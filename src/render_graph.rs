use std::collections::HashMap;
use std::ffi::CString;
use std::hash::Hash;
use std::marker::{Copy, PhantomData};
use std::ops::FnOnce;

use ash::vk;
use glam::{UVec2, UVec3};

use crate::command_buffer::CommandBuffer;
use crate::gpu_profiling::NamedProfiling;
use crate::render_device::buffer::{BufferView, BufferViewDesc};
use crate::render_device::{
    AccelerationStructure, Buffer, BufferDesc, RenderDevice, Texture, TextureDesc, TextureView,
    TextureViewDesc,
};
use crate::shader::{
    self, GraphicsDesc, Handle, Pipeline, PushConstantsBuilder, RayTracingDesc, ShaderDefinition,
    Shaders, ShadersConfig,
};

pub struct RGHandle<T> {
    id: usize,
    _phantom: PhantomData<T>,
}

impl<T> RGHandle<T> {
    fn new(id: usize) -> Self {
        RGHandle {
            id,
            _phantom: PhantomData::default(),
        }
    }

    fn null() -> Self {
        Self::new(usize::MAX)
    }
}

impl<T> Clone for RGHandle<T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            _phantom: PhantomData::default(),
        }
    }
}

impl<T> Copy for RGHandle<T> {}

impl<T> PartialEq for RGHandle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T> Eq for RGHandle<T> {}

impl<T> Hash for RGHandle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

pub struct RGTemporal<T> {
    id: usize,
    _phantom: PhantomData<T>,
}

impl<T> RGTemporal<T> {
    fn new(id: usize) -> Self {
        RGTemporal {
            id,
            _phantom: PhantomData::default(),
        }
    }
}

impl<T> Clone for RGTemporal<T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            _phantom: PhantomData::default(),
        }
    }
}

impl<T> Copy for RGTemporal<T> {}

#[derive(Clone, Copy)]
pub enum ColorLoadOp {
    Load,
    Clear(vk::ClearColorValue),
    DontCare,
}

#[derive(Clone, Copy)]
pub struct ColorTarget {
    pub tex: RGHandle<Texture>,
    pub layer: u32,
    pub load_op: ColorLoadOp,
}

impl Default for ColorTarget {
    fn default() -> Self {
        Self {
            tex: RGHandle::new(usize::MAX),
            layer: 0,
            load_op: ColorLoadOp::DontCare,
        }
    }
}

struct InternalColorTarget {
    view: RGHandle<TextureView>,
    load_op: ColorLoadOp,
}

#[derive(Clone, Copy)]
pub enum DepthLoadOp {
    Load,
    Clear(vk::ClearDepthStencilValue),
    DontCare,
}

#[derive(Clone, Copy)]
pub struct DepthStencilTarget {
    pub tex: RGHandle<Texture>,
    pub aspect: vk::ImageAspectFlags,
    pub load_op: DepthLoadOp,
    pub store_op: vk::AttachmentStoreOp,
}

impl Default for DepthStencilTarget {
    fn default() -> Self {
        Self {
            tex: RGHandle::null(),
            aspect: vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
            load_op: DepthLoadOp::DontCare,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
        }
    }
}

struct InternalDepthStencilTarget {
    view: RGHandle<TextureView>,
    load_op: DepthLoadOp,
    store_op: vk::AttachmentStoreOp,
}

struct GraphicsPassData {
    vertex_shader: Option<ShaderDefinition>,
    pixel_shader: Option<ShaderDefinition>,
    desc: GraphicsDesc,
    color_targets: Vec<InternalColorTarget>,
    depth_stencil: Option<InternalDepthStencilTarget>,
}

struct ComputePassData {
    shader: Option<ShaderDefinition>,
    group_count: UVec3,
}

struct RaytracingPassData {
    raygen_shader: Option<ShaderDefinition>,
    //miss_shader: Option<ShaderDefinition>,
    miss_shaders: Vec<ShaderDefinition>,
    chit_shader: Option<ShaderDefinition>,
    desc: RayTracingDesc,
    dimension: UVec3,
}

struct PresentData {
    texture: RGHandle<Texture>,
}

enum RenderPassType {
    Graphics(GraphicsPassData),
    Compute(ComputePassData),
    RayTracing(RaytracingPassData),
    Present(PresentData),
}

impl RenderPassType {
    fn to_bind_point(&self) -> Option<vk::PipelineBindPoint> {
        let bind_point = match self {
            RenderPassType::Graphics(_) => vk::PipelineBindPoint::GRAPHICS,
            RenderPassType::Compute(_) => vk::PipelineBindPoint::COMPUTE,
            RenderPassType::RayTracing(_) => vk::PipelineBindPoint::RAY_TRACING_KHR,
            RenderPassType::Present(_) => return None,
        };
        Some(bind_point)
    }

    fn is_graphics(&self) -> bool {
        match self {
            RenderPassType::Graphics(_) => true,
            _ => false,
        }
    }

    fn is_present(&self) -> bool {
        match self {
            RenderPassType::Present(_) => true,
            _ => false,
        }
    }
}

impl RenderPassType {
    fn gfx(&self) -> Option<&GraphicsPassData> {
        match self {
            RenderPassType::Graphics(gfx) => Some(gfx),
            _ => None,
        }
    }

    fn gfx_mut(&mut self) -> Option<&mut GraphicsPassData> {
        match self {
            RenderPassType::Graphics(gfx) => Some(gfx),
            _ => None,
        }
    }

    fn compute_mut(&mut self) -> Option<&mut ComputePassData> {
        match self {
            RenderPassType::Compute(compute) => Some(compute),
            _ => None,
        }
    }

    fn rt_mut(&mut self) -> Option<&mut RaytracingPassData> {
        match self {
            RenderPassType::RayTracing(rt) => Some(rt),
            _ => None,
        }
    }

    fn present(&self) -> Option<&PresentData> {
        match self {
            RenderPassType::Present(present) => Some(present),
            _ => None,
        }
    }
}

pub static DESCRIPTOR_SET_INDEX_UNUSED: u32 = u32::MAX;

// Lifetime marker 'render is required becaues RenderPass::render may (unmutably) reference the RenderScene (or other things that live in the render loop call)
pub struct RenderPass<'render> {
    name: String,
    ty: RenderPassType,
    descriptor_set_index: u32,
    textures: Vec<(&'static str, RGHandle<TextureView>)>,
    buffers: Vec<(&'static str, RGHandle<Buffer>)>,
    accel_structs: Vec<(&'static str, RGHandle<AccelerationStructure>)>,
    rw_textures: Vec<(&'static str, RGHandle<TextureView>)>,
    rw_buffers: Vec<(&'static str, RGHandle<Buffer>)>,
    rw_texel_buffers: Vec<(&'static str, RGHandle<BufferView>)>,
    push_constants: PushConstantsBuilder,
    set_layout_override: Option<(u32, vk::DescriptorSetLayout)>,
    pre_render: Option<Box<dyn 'render + FnOnce(&CommandBuffer, &Pipeline)>>,
    render: Option<Box<dyn 'render + FnOnce(&CommandBuffer, &Pipeline)>>,
}

impl RenderPass<'_> {
    fn new(name: &str, ty: RenderPassType) -> Self {
        RenderPass {
            name: String::from(name),
            ty,
            descriptor_set_index: 0,
            textures: Vec::new(),
            buffers: Vec::new(),
            accel_structs: Vec::new(),
            rw_textures: Vec::new(),
            rw_buffers: Vec::new(),
            rw_texel_buffers: Vec::new(),
            push_constants: PushConstantsBuilder::new(),
            set_layout_override: None,
            pre_render: None,
            render: None,
        }
    }
}

// Private interfaces for all pass builders, for implementing the public trait
// TODO any way in rust to actually make this private?
pub trait PrivatePassBuilderTrait<'render> {
    fn rg(&mut self) -> &mut RenderGraphBuilder<'render>;
    fn inner(&mut self) -> &mut RenderPass<'render>;
}

// Common interfaces for all pass builders
pub trait PassBuilderTrait<'render>: PrivatePassBuilderTrait<'render> {
    fn descriptor_set_index(&mut self, index: u32) -> &mut Self {
        self.inner().descriptor_set_index = index;
        self
    }

    /// Binding texture to per-pass descriptor set
    fn texture(&mut self, name: &'static str, texture: RGHandle<Texture>) -> &mut Self {
        let view = self.rg().create_texture_view(texture, None);
        self.inner().textures.push((name, view));
        self
    }

    /// Binding texture to per-pass descriptor set, as a TextureView
    fn texture_view(&mut self, name: &'static str, texture: RGHandle<TextureView>) -> &mut Self {
        self.inner().textures.push((name, texture));
        self
    }

    /// Binding rw texture to per-pass descriptor set
    fn rw_texture(&mut self, name: &'static str, texture: RGHandle<Texture>) -> &mut Self {
        let view = self.rg().create_texture_view(texture, None);
        self.inner().rw_textures.push((name, view));
        self
    }

    /// Binding rw texture to per-pass descriptor set, as a TextureView
    fn rw_texture_view(&mut self, name: &'static str, texture: RGHandle<TextureView>) -> &mut Self {
        self.inner().rw_textures.push((name, texture));
        self
    }

    fn buffer(&mut self, name: &'static str, buffer: RGHandle<Buffer>) -> &mut Self {
        self.inner().buffers.push((name, buffer));
        self
    }

    fn rw_buffer(&mut self, name: &'static str, buffer: RGHandle<Buffer>) -> &mut Self {
        self.inner().rw_buffers.push((name, buffer));
        self
    }

    fn rw_buffer_with_format(
        &mut self,
        name: &'static str,
        buffer: RGHandle<Buffer>,
        format: vk::Format,
    ) -> &mut Self {
        let view = self.rg().create_buffer_view(buffer, format);
        self.inner().rw_texel_buffers.push((name, view));
        self
    }

    // Binding acceleration structure to per-pass descriptor set
    fn accel_struct(
        &mut self,
        name: &'static str,
        accel_struct: RGHandle<AccelerationStructure>,
    ) -> &mut Self {
        self.inner().accel_structs.push((name, accel_struct));
        self
    }

    fn push_constant<T>(&mut self, value: &T) -> &mut Self
    where
        T: Copy + Sized,
    {
        self.inner().push_constants.push_inplace::<T>(value);
        self
    }

    fn set_layout_override(
        &mut self,
        index: u32,
        set_layout: vk::DescriptorSetLayout,
    ) -> &mut Self {
        self.inner()
            .set_layout_override
            .replace((index, set_layout));
        self
    }

    // Pre-Render function, call before the render pass instance
    fn pre_render<F>(&mut self, f: F) -> &mut Self
    where
        F: 'render + FnOnce(&CommandBuffer, &Pipeline),
    {
        self.inner().pre_render = Some(Box::new(f));
        self
    }

    // Render function for graphics pass
    fn render<F>(&mut self, f: F) -> &mut Self
    where
        F: 'render + FnOnce(&CommandBuffer, &Pipeline),
    {
        self.inner().render = Some(Box::new(f));
        self
    }
}

macro_rules! define_pass_builder {
    ($pass_builder:ident) => {
        // Declare
        pub struct $pass_builder<'a, 'render> {
            inner: Option<RenderPass<'render>>,
            render_graph: &'a mut RenderGraphBuilder<'render>,
        }

        // Implement methods
        impl<'render> PrivatePassBuilderTrait<'render> for $pass_builder<'_, 'render> {
            fn rg(&mut self) -> &mut RenderGraphBuilder<'render> {
                self.render_graph
            }

            fn inner(&mut self) -> &mut RenderPass<'render> {
                self.inner.as_mut().unwrap()
            }
        }

        impl<'render> PassBuilderTrait<'render> for $pass_builder<'_, 'render> {}

        impl Drop for $pass_builder<'_, '_> {
            fn drop(&mut self) {
                // Add the pass to render graph automatically after last method call (drop form rg::new_*())
                if self.inner.is_some() {
                    let inner = self.inner.take();
                    self.render_graph.add_pass(inner.unwrap());
                }
            }
        }
    };
}

define_pass_builder!(GraphicsPassBuilder);

impl<'a, 'render> GraphicsPassBuilder<'a, 'render> {
    fn new(render_graph: &'a mut RenderGraphBuilder<'render>, name: &str) -> Self {
        let ty = RenderPassType::Graphics(GraphicsPassData {
            vertex_shader: None,
            pixel_shader: None,
            desc: GraphicsDesc::default(),
            color_targets: Vec::new(),
            depth_stencil: None,
        });
        let pass = RenderPass::new(name, ty);
        Self {
            inner: Some(pass),
            render_graph,
        }
    }

    fn gfx(&mut self) -> &mut GraphicsPassData {
        self.inner.as_mut().unwrap().ty.gfx_mut().unwrap()
    }

    pub fn vertex_shader_with_ep(
        &mut self,
        path: &'static str,
        entry_point: &'static str,
    ) -> &mut Self {
        self.gfx().vertex_shader.replace(ShaderDefinition {
            virtual_path: path,
            entry_point: entry_point,
            stage: shader::ShaderStage::Vert,
        });
        self
    }

    pub fn pixel_shader_with_ep(
        &mut self,
        path: &'static str,
        entry_point: &'static str,
    ) -> &mut Self {
        self.gfx().pixel_shader.replace(ShaderDefinition {
            virtual_path: path,
            entry_point: entry_point,
            stage: shader::ShaderStage::Frag,
        });
        self
    }

    pub fn color_targets(&mut self, rts: &[ColorTarget]) -> &mut Self {
        assert!(rts.len() < u8::MAX as usize);
        assert!(rts.len() <= self.gfx().desc.color_attachments.len());

        self.gfx().desc.color_attachment_count = rts.len() as u8;

        self.gfx().color_targets.clear();

        for i in 0..rts.len() {
            let format = self.render_graph.get_texture_desc(rts[i].tex).format;
            self.gfx().desc.color_attachments[i] = format;

            let view = self.render_graph.create_texture_view(
                rts[i].tex,
                Some(TextureViewDesc {
                    view_type: vk::ImageViewType::TYPE_2D,
                    format,
                    aspect: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: rts[i].layer,
                    layer_count: 1,
                    ..Default::default()
                }),
            );

            self.gfx().color_targets.push(InternalColorTarget {
                view: view,
                load_op: rts[i].load_op,
            })
        }

        self
    }

    pub fn depth_stencil(&mut self, ds: DepthStencilTarget) -> &mut Self {
        let format = self.render_graph.get_texture_desc(ds.tex).format;

        // TODO maybe only one is needed? (sometimes)
        self.gfx().desc.depth_attachment = Some(format);
        self.gfx().desc.stencil_attachment = Some(format);

        let view = self.rg().create_texture_view(
            ds.tex,
            Some(TextureViewDesc {
                view_type: vk::ImageViewType::TYPE_2D,
                format,
                aspect: ds.aspect,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            }),
        );
        self.gfx().depth_stencil = Some(InternalDepthStencilTarget {
            view,
            load_op: ds.load_op,
            store_op: ds.store_op,
        });
        self
    }

    pub fn blend_enabled(&mut self, enabled: bool) -> &mut Self {
        self.gfx().desc.blend_enabled = enabled;
        self
    }
}

define_pass_builder!(ComputePassBuilder);

impl<'a, 'render> ComputePassBuilder<'a, 'render> {
    pub fn new(render_graph: &'a mut RenderGraphBuilder<'render>, name: &str) -> Self {
        let ty = RenderPassType::Compute(ComputePassData {
            shader: None,
            group_count: UVec3::new(0, 0, 0),
        });
        let pass = RenderPass::new(name, ty);
        Self {
            inner: Some(pass),
            render_graph,
        }
    }

    fn compute(&mut self) -> &mut ComputePassData {
        self.inner.as_mut().unwrap().ty.compute_mut().unwrap()
    }

    pub fn compute_shader(&mut self, path: &'static str) -> &mut Self {
        self.compute().shader.replace(ShaderDefinition {
            virtual_path: path,
            entry_point: "main",
            stage: shader::ShaderStage::Compute,
        });
        self
    }

    pub fn group_count(&mut self, x: u32, y: u32, z: u32) -> &mut Self {
        self.compute().group_count = UVec3::new(x, y, z);
        self
    }

    pub fn group_count_uvec3(&mut self, group_count: UVec3) -> &mut Self {
        self.compute().group_count = group_count;
        self
    }

    pub fn group_count_uvec2(&mut self, group_count: UVec2) -> &mut Self {
        self.compute().group_count = group_count.extend(1);
        self
    }
}

define_pass_builder!(RaytracingPassBuilder);

impl<'a, 'render> RaytracingPassBuilder<'a, 'render> {
    pub fn new(render_graph: &'a mut RenderGraphBuilder<'render>, name: &str) -> Self {
        let ty = RenderPassType::RayTracing(RaytracingPassData {
            raygen_shader: None,
            miss_shaders: Vec::new(),
            chit_shader: None,
            desc: RayTracingDesc {
                ray_recursiion_depth: 1,
            },
            dimension: UVec3::new(0, 0, 0),
        });
        let pass = RenderPass::new(name, ty);
        Self {
            inner: Some(pass),
            render_graph,
        }
    }

    fn rt(&mut self) -> &mut RaytracingPassData {
        self.inner.as_mut().unwrap().ty.rt_mut().unwrap()
    }

    pub fn raygen_shader(&mut self, path: &'static str) -> &mut Self {
        self.rt().raygen_shader.replace(ShaderDefinition {
            virtual_path: path,
            entry_point: "main",
            stage: shader::ShaderStage::RayGen,
        });
        self
    }

    pub fn raygen_shader_with_ep(
        &mut self,
        path: &'static str,
        entry_point: &'static str,
    ) -> &mut Self {
        self.rt().raygen_shader.replace(ShaderDefinition {
            virtual_path: path,
            entry_point,
            stage: shader::ShaderStage::RayGen,
        });
        self
    }

    pub fn miss_shader(&mut self, path: &'static str) -> &mut Self {
        self.rt().miss_shaders.clear();
        self.rt().miss_shaders.push(ShaderDefinition {
            virtual_path: path,
            entry_point: "main",
            stage: shader::ShaderStage::Miss,
        });
        self
    }

    pub fn miss_shaders(&mut self, paths: &[&'static str]) -> &mut Self {
        self.rt().miss_shaders = paths
            .iter()
            .map(|path| ShaderDefinition {
                virtual_path: path,
                entry_point: "main",
                stage: shader::ShaderStage::Miss,
            })
            .collect();
        self
    }

    pub fn miss_shader_with_ep(
        &mut self,
        path: &'static str,
        entry_point: &'static str,
    ) -> &mut Self {
        self.rt().miss_shaders.clear();
        self.rt().miss_shaders.push(ShaderDefinition {
            virtual_path: path,
            entry_point,
            stage: shader::ShaderStage::Miss,
        });
        self
    }

    pub fn closest_hit_shader(&mut self, path: &'static str) -> &mut Self {
        self.rt().chit_shader.replace(ShaderDefinition {
            virtual_path: path,
            entry_point: "main",
            stage: shader::ShaderStage::ClosestHit,
        });
        self
    }

    pub fn closest_hit_shader_with_ep(
        &mut self,
        path: &'static str,
        entry_point: &'static str,
    ) -> &mut Self {
        self.rt().chit_shader.replace(ShaderDefinition {
            virtual_path: path,
            entry_point: entry_point,
            stage: shader::ShaderStage::ClosestHit,
        });
        self
    }

    pub fn max_recursion_depth(&mut self, depth: u32) -> &mut Self {
        self.rt().desc.ray_recursiion_depth = depth;
        self
    }

    pub fn dimension(&mut self, width: u32, height: u32, depth: u32) -> &mut Self {
        self.rt().dimension = UVec3::new(width, height, depth);
        self
    }
}

struct ResPool<K, V>
where
    K: Eq + Hash,
{
    map: HashMap<K, Vec<V>>,
}

impl<K, V> ResPool<K, V>
where
    K: Eq + Hash,
{
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    pub fn push(&mut self, key: K, v: V) {
        self.map.entry(key).or_insert_with(Vec::new).push(v);
    }

    pub fn pop(&mut self, key: &K) -> Option<V> {
        let pooled = self.map.get_mut(key);
        if let Some(pooled) = pooled {
            pooled.pop()
        } else {
            None
        }
    }
}

static MAX_NUM_MISS_SHADERS: u32 = 2;
//static MAX_NUM_MISS_SHADERS: u32 = 1;

// An internal resource type for ray tracing passes.
// This kind of passes have just a few shander handles, and no extra data in shader binding tables.
struct PassShaderBindingTable {
    sbt: Buffer,
    raygen_region: vk::StridedDeviceAddressRegionKHR,
    miss_region: vk::StridedDeviceAddressRegionKHR,
    hit_region: vk::StridedDeviceAddressRegionKHR,
}

impl PassShaderBindingTable {
    fn new(rd: &RenderDevice) -> Self {
        // Device properties
        let handle_size = rd.shader_group_handle_size() as u64;
        let handle_alignment = rd.shader_group_handle_alignment() as u64;
        let base_alignment = rd.shader_group_base_alignment() as u64;

        // routine to align address/stride
        let base_align =
            |offset: u64| ((offset + base_alignment - 1) / base_alignment) * base_alignment;
        let handle_align =
            |stride: u64| ((stride + handle_alignment - 1) / handle_alignment) * handle_alignment;

        // Sub allocate offsets
        let raygen_size = handle_size;
        let miss_offset = base_align(raygen_size);
        let miss_stride = handle_align(handle_size);
        let miss_size = handle_size + miss_stride * (MAX_NUM_MISS_SHADERS - 1) as u64;
        let hit_offset = base_align(miss_offset + miss_size);
        let hit_stride = handle_align(handle_size);
        let hit_size = handle_size;

        let buffer_size = hit_offset + hit_size;

        let sbt = rd
            .create_buffer(BufferDesc {
                size: buffer_size,
                usage: vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                memory_property: vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT,
            })
            .unwrap();

        let raygen_region = vk::StridedDeviceAddressRegionKHR {
            device_address: sbt.device_address.unwrap(),
            stride: raygen_size, // required by spec
            size: raygen_size,
        };

        let miss_region = vk::StridedDeviceAddressRegionKHR {
            device_address: sbt.device_address.unwrap() + miss_offset,
            stride: miss_stride,
            size: miss_size,
        };

        let hit_region = vk::StridedDeviceAddressRegionKHR {
            device_address: sbt.device_address.unwrap() + hit_offset,
            stride: hit_stride,
            size: hit_size,
        };

        Self {
            sbt,
            raygen_region,
            miss_region,
            hit_region,
        }
    }

    fn update_shader_group_handles(
        &mut self,
        rd: &RenderDevice,
        pipeline: &Pipeline,
        num_miss: u32,
        has_hit: bool,
    ) {
        assert!(num_miss > 0);
        assert!(num_miss <= MAX_NUM_MISS_SHADERS);

        // TODO check shader change
        let group_count = 1 + num_miss + if has_hit { 1 } else { 0 };
        let handle_data = rd.get_shader_group_handles(pipeline.handle, 0, group_count);

        // procedure to copy a shader handle
        let handle_size = rd.shader_group_handle_size();
        let copy_handle = |handle_index: u32, dst_offset: u64| unsafe {
            let src = std::slice::from_raw_parts(
                handle_data
                    .as_ptr()
                    .offset((handle_size * handle_index) as isize),
                handle_size as usize,
            );
            let dst = std::slice::from_raw_parts_mut(
                self.sbt.data.offset(dst_offset as isize),
                handle_size as usize,
            );
            dst.copy_from_slice(src);
        };
        let empty_handle = |dst_offset: u64| unsafe {
            let dst = std::slice::from_raw_parts_mut(
                self.sbt.data.offset(dst_offset as isize),
                handle_size as usize,
            );
            dst.fill(0);
        };

        // Copy shader handle to the buffer
        let base_device_address = self.sbt.device_address.unwrap();
        // raygen
        let raygen_offset = self.raygen_region.device_address - base_device_address;
        copy_handle(0, raygen_offset);
        // miss
        // TODO this one-by-one copy can be just memcpy when shader_group_handle_size == shader_group_handle_alignment
        let miss_offset = self.miss_region.device_address - base_device_address;
        for i in 0..num_miss {
            copy_handle(1 + i, miss_offset + self.miss_region.stride * i as u64);
        }
        // clear the remaining handle slots, for sanity
        for i in num_miss..MAX_NUM_MISS_SHADERS {
            empty_handle(miss_offset + self.miss_region.stride * (i + num_miss) as u64);
        }
        // hit
        if has_hit {
            let hit_offset = self.hit_region.device_address - base_device_address;
            copy_handle(1 + num_miss, hit_offset);
        }
    }
}

const RG_MAX_SET: u32 = 1024;

pub struct RenderGraphCache {
    // Resused VK stuffs
    vk_descriptor_pool: vk::DescriptorPool,

    // buffered VK objects
    free_vk_descriptor_sets: Vec<vk::DescriptorSet>,

    // Temporal Resources
    temporal_textures: HashMap<usize, Texture>,
    temporal_buffers: HashMap<usize, Buffer>,
    next_temporal_id: usize,

    // Resource pool
    texture_pool: ResPool<TextureDesc, Texture>,
    texture_view_pool: ResPool<(Texture, TextureViewDesc), TextureView>,
    buffer_pool: ResPool<BufferDesc, Buffer>,
    buffer_view_pool: ResPool<(Buffer, BufferViewDesc), BufferView>,
    sbt_pool: ResPool<u32, PassShaderBindingTable>,

    // Accumulated pass profilng info
    // TODO should just pass in
    pub pass_profiling: NamedProfiling,
}

impl RenderGraphCache {
    pub fn new(rd: &RenderDevice) -> Self {
        let vk_descriptor_pool = rd.create_descriptor_pool(
            vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
            RG_MAX_SET,
            &[vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 1024,
            }],
        );
        Self {
            vk_descriptor_pool,
            free_vk_descriptor_sets: Vec::new(),
            temporal_textures: HashMap::new(),
            temporal_buffers: HashMap::new(),
            next_temporal_id: 0,
            texture_pool: ResPool::new(),
            texture_view_pool: ResPool::new(),
            buffer_pool: ResPool::new(),
            buffer_view_pool: ResPool::new(),
            sbt_pool: ResPool::new(),
            pass_profiling: NamedProfiling::new(rd),
        }
    }

    fn allocate_dessriptor_set(
        &mut self,
        rd: &RenderDevice,
        set_layout: vk::DescriptorSetLayout,
    ) -> vk::DescriptorSet {
        let layouts = [set_layout];
        let create_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.vk_descriptor_pool)
            .set_layouts(&layouts);
        unsafe {
            match rd.device.allocate_descriptor_sets(&create_info) {
                Ok(sets) => sets[0],
                Err(e) => {
                    panic!("Failed to allocate descriptor set: {:?}", e);
                }
            }
        }
    }

    fn release_descriptor_set(&mut self, rd: &RenderDevice, descriptor_set: vk::DescriptorSet) {
        self.free_vk_descriptor_sets.push(descriptor_set);

        // TODO free after command buffer is done (fence)
        let buffer_limit = (RG_MAX_SET / 2) as usize;
        let release_heuristic = buffer_limit / 2;
        if self.free_vk_descriptor_sets.len() > buffer_limit {
            let old_sets = &self.free_vk_descriptor_sets[0..release_heuristic];
            unsafe {
                rd.device
                    .free_descriptor_sets(self.vk_descriptor_pool, &old_sets)
                    .expect("Failed to free descriptor set");
            }
            self.free_vk_descriptor_sets.drain(0..release_heuristic);
        }
    }
}

struct RenderGraphExecuteContext {
    // Created Resources
    pub textures: Vec<Option<Texture>>,          // by handle::id
    pub texture_views: Vec<Option<TextureView>>, // by handle::id
    pub buffers: Vec<Option<Buffer>>,            // by handle::id
    pub buffer_views: Vec<Option<BufferView>>,   // by handle::id

    // Per-pass created pipeline
    pub pipelines: Vec<Handle<Pipeline>>,

    // Per-pass descriptor set
    pub descriptor_sets: Vec<vk::DescriptorSet>, // by pass index

    // Per-pass shader binding tables
    pub shader_binding_tables: Vec<Option<PassShaderBindingTable>>, // by pass index
}

impl RenderGraphExecuteContext {
    pub fn new() -> Self {
        Self {
            textures: Vec::new(),
            texture_views: Vec::new(),
            buffers: Vec::new(),
            buffer_views: Vec::new(),
            pipelines: Vec::new(),
            descriptor_sets: Vec::new(),
            shader_binding_tables: Vec::new(),
        }
    }
}

enum RenderResource<V, E> {
    Virtual(V),      // a.k.a. Transient Resources
    Temporal(usize), // Index to the cache
    External(E),
}

struct VirtualTextureView {
    pub texture: RGHandle<Texture>,
    pub desc: Option<TextureViewDesc>, // if none, use ::auto::(texture::desc)
}

struct VirtualBufferView {
    pub buffer: RGHandle<Buffer>,
    pub desc: BufferViewDesc,
}

pub enum ResTypeEnum {
    Texture,
    Buffer,
}

pub trait ResType {
    fn get_enum() -> ResTypeEnum;
}

impl ResType for Texture {
    fn get_enum() -> ResTypeEnum {
        ResTypeEnum::Texture
    }
}

impl ResType for Buffer {
    fn get_enum() -> ResTypeEnum {
        ResTypeEnum::Buffer
    }
}

// A Render Graph to handle resource transitions automatically
pub struct RenderGraphBuilder<'render> {
    passes: Vec<RenderPass<'render>>,

    shader_config: shader::ShadersConfig,
    cache: RenderGraphCache,

    // Descriptor sets that would be bound for all passes
    // Exist for ergonomics reason; descriptors set like per-frame stuffs can be specify as this.
    global_descriptor_sets: Vec<(u32, vk::DescriptorSet)>,

    transient_to_temporal_textures: HashMap<RGHandle<Texture>, RGTemporal<Texture>>,
    transient_to_temporal_buffers: HashMap<RGHandle<Buffer>, RGTemporal<Buffer>>,

    // Array indexed by RGHandle
    textures: Vec<RenderResource<TextureDesc, Texture>>,
    texture_views: Vec<RenderResource<VirtualTextureView, TextureView>>,
    buffers: Vec<RenderResource<BufferDesc, Buffer>>,
    buffer_views: Vec<RenderResource<VirtualBufferView, BufferView>>,
    accel_structs: Vec<RenderResource<(), AccelerationStructure>>,

    hack_frame_index: u32,
}

// Private stuff
impl<'a> RenderGraphBuilder<'a> {
    fn add_pass(&mut self, pass: RenderPass<'a>) {
        self.passes.push(pass);
    }

    /*
    fn is_virtual<T>(&self, handle: RGHandle<T>) -> bool
    where
        T: ResType,
    {
        match T::get_enum() {
            ResTypeEnum::Texture => match self.textures.get(handle.id) {
                Some(RenderResource::Virtual(_)) => true,
                _ => false,
            },
            ResTypeEnum::Buffer => match self.buffers.get(handle.id) {
                Some(RenderResource::Virtual(_)) => true,
                _ => false,
            },
        }
    }
    */

    fn is_external<T>(&self, handle: RGHandle<T>) -> bool
    where
        T: ResType,
    {
        match T::get_enum() {
            ResTypeEnum::Texture => match self.textures.get(handle.id) {
                Some(RenderResource::External(_)) => true,
                _ => false,
            },
            ResTypeEnum::Buffer => match self.buffers.get(handle.id) {
                Some(RenderResource::External(_)) => true,
                _ => false,
            },
        }
    }

    fn get_temporal_id<T>(&self, handle: RGHandle<T>) -> Option<usize>
    where
        T: ResType,
    {
        match T::get_enum() {
            ResTypeEnum::Texture => match self.textures.get(handle.id) {
                Some(RenderResource::Temporal(id)) => Some(*id),
                _ => None,
            },
            ResTypeEnum::Buffer => match self.buffers.get(handle.id) {
                Some(RenderResource::Temporal(id)) => Some(*id),
                _ => None,
            },
        }
    }

    fn find_converted_temporal<T>(&self, temporal_handle: RGTemporal<T>) -> Option<RGHandle<T>>
    where
        T: ResType,
    {
        let temporal_id = temporal_handle.id;

        match T::get_enum() {
            ResTypeEnum::Texture => {
                for i in 0..self.textures.len() {
                    if let RenderResource::Temporal(id) = &self.textures[i] {
                        if *id == temporal_id {
                            return Some(RGHandle::new(i));
                        }
                    }
                }
            }
            ResTypeEnum::Buffer => {
                for i in 0..self.buffers.len() {
                    if let RenderResource::Temporal(id) = &self.buffers[i] {
                        if *id == temporal_id {
                            return Some(RGHandle::new(i));
                        }
                    }
                }
            }
        };

        None
    }

    fn find_converted_to_temporal<T>(&self, handle: RGHandle<T>) -> Option<RGTemporal<T>>
    where
        T: ResType,
    {
        let handle_id = handle.id;
        match T::get_enum() {
            ResTypeEnum::Texture => self
                .transient_to_temporal_textures
                .get(&RGHandle::new(handle_id))
                .map(|t| RGTemporal::new(t.id)),
            ResTypeEnum::Buffer => self
                .transient_to_temporal_buffers
                .get(&RGHandle::new(handle_id))
                .map(|t| RGTemporal::new(t.id)),
        }
    }
}

// Interface
impl<'a> RenderGraphBuilder<'a> {
    pub fn new(cache: RenderGraphCache, shader_config: ShadersConfig) -> Self {
        Self {
            passes: Vec::new(),

            shader_config,
            cache,

            global_descriptor_sets: Vec::new(),

            transient_to_temporal_textures: HashMap::new(),
            transient_to_temporal_buffers: HashMap::new(),

            textures: Vec::new(),
            texture_views: Vec::new(),
            buffers: Vec::new(),
            buffer_views: Vec::new(),
            accel_structs: Vec::new(),

            hack_frame_index: 0,
        }
    }

    pub fn set_frame_index(&mut self, frame_index: u32) {
        self.hack_frame_index = frame_index;
    }

    pub fn create_texutre(&mut self, desc: TextureDesc) -> RGHandle<Texture> {
        let id = self.textures.len();
        self.textures.push(RenderResource::Virtual(desc));
        RGHandle::new(id)
    }

    pub fn create_texture_view(
        &mut self,
        texture: RGHandle<Texture>,
        desc: Option<TextureViewDesc>,
    ) -> RGHandle<TextureView> {
        // Check if already created
        for handle_id in 0..self.texture_views.len() {
            if let RenderResource::Virtual(virtual_view) = &self.texture_views[handle_id] {
                if virtual_view.desc == desc && virtual_view.texture == texture {
                    return RGHandle::new(handle_id);
                }
            }
        }

        let id = self.texture_views.len();
        self.texture_views
            .push(RenderResource::Virtual(VirtualTextureView {
                texture,
                desc,
            }));
        RGHandle::new(id)
    }

    pub fn register_texture(&mut self, texture: Texture) -> RGHandle<Texture> {
        // Check if already registered
        // TOOD make this routine part of self.textures
        for handle_id in 0..self.textures.len() {
            let texture_resource = &self.textures[handle_id];
            if let RenderResource::External(prev_tex) = texture_resource {
                if *prev_tex == texture {
                    return RGHandle::new(handle_id);
                }
            }
        }

        // Register the texture
        let id = self.textures.len();
        self.textures.push(RenderResource::External(texture));
        RGHandle::new(id)
    }

    pub fn register_texture_view(&mut self, texture_view: TextureView) -> RGHandle<TextureView> {
        // Check is already registered
        for handle_id in 0..self.texture_views.len() {
            let view_resource = &self.texture_views[handle_id];
            if let RenderResource::External(prev_texture_view) = view_resource {
                if *prev_texture_view == texture_view {
                    return RGHandle::new(handle_id);
                }
            }
        }

        // Register underlying texture (for dependecy tracking)
        self.register_texture(texture_view.texture);

        // Register the view
        let id = self.texture_views.len();
        self.texture_views
            .push(RenderResource::External(texture_view));
        RGHandle::new(id)
    }

    pub fn create_buffer(&mut self, desc: BufferDesc) -> RGHandle<Buffer> {
        let id = self.buffers.len();
        self.buffers.push(RenderResource::Virtual(desc));
        RGHandle::new(id)
    }

    pub fn create_buffer_view(
        &mut self,
        buffer: RGHandle<Buffer>,
        desc: BufferViewDesc,
    ) -> RGHandle<BufferView> {
        // Check if already created
        for handle_id in 0..self.buffer_views.len() {
            if let RenderResource::Virtual(virtual_view) = &self.buffer_views[handle_id] {
                if virtual_view.desc == desc && virtual_view.buffer == buffer {
                    return RGHandle::new(handle_id);
                }
            }
        }

        let id = self.buffer_views.len();
        self.buffer_views
            .push(RenderResource::Virtual(VirtualBufferView { buffer, desc }));
        RGHandle::new(id)
    }

    pub fn register_buffer(&mut self, buffer: Buffer) -> RGHandle<Buffer> {
        // Check if already registered
        for handle_id in 0..self.buffers.len() {
            let res = &self.buffers[handle_id];
            if let RenderResource::External(prev_buf) = res {
                if *prev_buf == buffer {
                    return RGHandle::new(handle_id);
                }
            }
        }

        // Register the buffer
        let id = self.buffers.len();
        self.buffers.push(RenderResource::External(buffer));
        RGHandle::new(id)
    }

    pub fn register_accel_struct(
        &mut self,
        accel_struct: AccelerationStructure,
    ) -> RGHandle<AccelerationStructure> {
        // Check is already registered
        for handle_id in 0..self.accel_structs.len() {
            let view_resource = &self.accel_structs[handle_id];
            if let RenderResource::External(prev) = view_resource {
                if *prev == accel_struct {
                    return RGHandle::new(handle_id);
                }
            }
        }

        // Register the underlying buffer
        // TODO

        // Register the accel struct
        let id = self.accel_structs.len();
        self.accel_structs
            .push(RenderResource::External(accel_struct));
        RGHandle::new(id)
    }

    // Convert a transient resource to a temporal one (content is kept through frames)
    pub fn convert_to_temporal<T>(&mut self, handle: RGHandle<T>) -> RGTemporal<T>
    where
        T: ResType,
    {
        // check if already converted
        if let Some(temporal) = self.find_converted_to_temporal(handle) {
            return temporal;
        }

        // Validate: not valid for external resource
        assert!(!self.is_external(handle));

        // all type using same temporal id scope... :)
        let temporal_id;
        if let Some(prev_temporal_id) = self.get_temporal_id(handle) {
            // reused the temporal id
            temporal_id = prev_temporal_id
        } else {
            // alocate a new one
            temporal_id = self.cache.next_temporal_id;
            self.cache.next_temporal_id += 1;
        };

        // record for later caching
        match T::get_enum() {
            ResTypeEnum::Texture => {
                self.transient_to_temporal_textures
                    .insert(RGHandle::new(handle.id), RGTemporal::new(temporal_id));
            }
            ResTypeEnum::Buffer => {
                self.transient_to_temporal_buffers
                    .insert(RGHandle::new(handle.id), RGTemporal::new(temporal_id));
            }
        }

        RGTemporal::<T>::new(temporal_id)
    }

    // Convert a temporal resource to a transient one (content is discarded after last usage in this frame)
    pub fn convert_to_transient<T>(&mut self, temporal_handle: RGTemporal<T>) -> RGHandle<T>
    where
        T: ResType,
    {
        // Check if already registered
        if let Some(handle) = self.find_converted_temporal(temporal_handle) {
            return handle;
        }

        // Register the texture
        let id = match T::get_enum() {
            ResTypeEnum::Texture => {
                let id = self.textures.len();
                self.textures
                    .push(RenderResource::Temporal(temporal_handle.id));
                assert!(!self
                    .transient_to_temporal_textures
                    .contains_key(&RGHandle::new(id)));
                id
            }
            ResTypeEnum::Buffer => {
                let id = self.buffers.len();
                self.buffers
                    .push(RenderResource::Temporal(temporal_handle.id));
                assert!(!self
                    .transient_to_temporal_buffers
                    .contains_key(&RGHandle::new(id)));
                id
            }
        };

        RGHandle::new(id)
    }

    pub fn get_texture_desc(&self, texture: RGHandle<Texture>) -> &TextureDesc {
        match &self.textures[texture.id] {
            RenderResource::Virtual(desc) => desc,
            RenderResource::Temporal(_) => todo!("Temporal resources are in cache"),
            RenderResource::External(texture) => &texture.desc,
        }
    }

    pub fn get_texture_desc_from_view(&self, texture_view: RGHandle<TextureView>) -> &TextureDesc {
        match &self.texture_views[texture_view.id] {
            RenderResource::Virtual(virtual_view) => self.get_texture_desc(virtual_view.texture),
            RenderResource::Temporal(temporal_id) => {
                &self.cache.temporal_textures[temporal_id].desc
            }
            RenderResource::External(texture_view) => &texture_view.texture.desc,
        }
    }

    pub fn get_buffer_desc(&self, buffer: RGHandle<Buffer>) -> &BufferDesc {
        match &self.buffers[buffer.id] {
            RenderResource::Virtual(desc) => desc,
            RenderResource::Temporal(temporal_id) => &self.cache.temporal_buffers[temporal_id].desc,
            RenderResource::External(buffer) => &buffer.desc,
        }
    }

    pub fn add_global_descriptor_sets(&mut self, sets: &[(u32, vk::DescriptorSet)]) {
        for (index, _set) in sets {
            if self
                .global_descriptor_sets
                .iter()
                .find(|(i, _)| i == index)
                .is_some()
            {
                println!("Warning: global descriptor set {} is overrided", index);
                continue;
            }
        }

        self.global_descriptor_sets.extend_from_slice(sets);
    }

    pub fn new_graphics<'tmp>(&'tmp mut self, name: &str) -> GraphicsPassBuilder<'tmp, 'a> {
        GraphicsPassBuilder::new(self, name)
    }

    pub fn new_compute<'tmp>(&'tmp mut self, name: &str) -> ComputePassBuilder<'tmp, 'a> {
        ComputePassBuilder::new(self, name)
    }

    pub fn new_raytracing<'tmp>(&'tmp mut self, name: &str) -> RaytracingPassBuilder<'tmp, 'a> {
        RaytracingPassBuilder::new(self, name)
    }

    pub fn present(&mut self, texture: RGHandle<Texture>) {
        let present_data = PresentData { texture };
        let pass = RenderPass::new("Present", RenderPassType::Present(present_data));
        self.passes.push(pass);
    }
}

// Internal Methods
impl RenderGraphBuilder<'_> {
    // Assume execute context is already populated
    #[inline]
    fn get_texture(&self, ctx: &RenderGraphExecuteContext, handle: RGHandle<Texture>) -> Texture {
        let resource = &self.textures[handle.id];
        match resource {
            RenderResource::Virtual(_) => ctx.textures[handle.id].unwrap(),
            RenderResource::Temporal(_) => ctx.textures[handle.id].unwrap(),
            RenderResource::External(texture) => *texture,
        }
    }

    // Assume execute context is already populated
    #[inline]
    fn get_texture_view(
        &self,
        ctx: &RenderGraphExecuteContext,
        handle: RGHandle<TextureView>,
    ) -> TextureView {
        let resource = &self.texture_views[handle.id];
        match resource {
            RenderResource::Virtual(_) => ctx.texture_views[handle.id].unwrap(),
            RenderResource::Temporal(_) => ctx.texture_views[handle.id].unwrap(),
            RenderResource::External(view) => *view,
        }
    }

    #[inline]
    fn get_buffer(&self, ctx: &RenderGraphExecuteContext, handle: RGHandle<Buffer>) -> Buffer {
        match &self.buffers[handle.id] {
            RenderResource::Virtual(_) => ctx.buffers[handle.id].unwrap(),
            RenderResource::Temporal(_) => ctx.buffers[handle.id].unwrap(),
            RenderResource::External(buffer) => *buffer,
        }
    }

    #[inline]
    fn get_buffer_view(
        &self,
        ctx: &RenderGraphExecuteContext,
        handle: RGHandle<BufferView>,
    ) -> BufferView {
        match &self.buffer_views[handle.id] {
            RenderResource::Virtual(_) => ctx.buffer_views[handle.id].unwrap(),
            RenderResource::Temporal(_) => ctx.buffer_views[handle.id].unwrap(),
            RenderResource::External(view) => *view,
        }
    }

    #[inline]
    fn get_accel_struct(&self, handle: RGHandle<AccelerationStructure>) -> AccelerationStructure {
        let accel_struct = &self.accel_structs[handle.id];
        match accel_struct {
            RenderResource::Virtual(_) => unimplemented!(),
            RenderResource::Temporal(_) => unimplemented!(),
            RenderResource::External(accel_struct) => *accel_struct,
        }
    }

    fn begin_graphics(
        &self,
        ctx: &RenderGraphExecuteContext,
        command_buffer: &CommandBuffer,
        pass: &RenderPass,
    ) {
        let gfx = pass.ty.gfx().unwrap();

        let size = if gfx.color_targets.len() > 0 {
            self.get_texture_desc_from_view(gfx.color_targets[0].view)
                .size_2d()
        } else if let Some(ds) = &gfx.depth_stencil {
            self.get_texture_desc_from_view(ds.view).size_2d()
        } else {
            panic!();
        };

        let color_attachments: Vec<_> = gfx
            .color_targets
            .iter()
            .map(|target| {
                let image_view = self.get_texture_view(ctx, target.view).image_view;

                let mut builder = vk::RenderingAttachmentInfo::builder()
                    .image_view(image_view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .store_op(vk::AttachmentStoreOp::STORE);

                builder = match target.load_op {
                    ColorLoadOp::Load => builder.load_op(vk::AttachmentLoadOp::LOAD),
                    ColorLoadOp::Clear(clear_value) => builder
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .clear_value(vk::ClearValue { color: clear_value }),
                    ColorLoadOp::DontCare => builder.load_op(vk::AttachmentLoadOp::DONT_CARE),
                };

                builder.build()
            })
            .collect();

        let depth_attachment = gfx.depth_stencil.as_ref().map(|target| {
            let image_view = self.get_texture_view(ctx, target.view).image_view;

            let mut builder = vk::RenderingAttachmentInfoKHR::builder()
                .image_view(image_view)
                .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .store_op(target.store_op);

            builder = match target.load_op {
                DepthLoadOp::Load => builder.load_op(vk::AttachmentLoadOp::LOAD),
                DepthLoadOp::Clear(clear_value) => builder
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .clear_value(vk::ClearValue {
                        depth_stencil: clear_value,
                    }),
                DepthLoadOp::DontCare => builder.load_op(vk::AttachmentLoadOp::DONT_CARE),
            };

            builder.build()
        });

        let mut rendering_info = vk::RenderingInfo::builder()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: size,
            })
            .layer_count(1)
            .view_mask(0)
            .color_attachments(&color_attachments);
        if depth_attachment.is_some() {
            // TODO maybe set also/only stencil_attachment in some case?
            rendering_info = rendering_info.depth_attachment(depth_attachment.as_ref().unwrap());
        };

        command_buffer.begin_rendering(&rendering_info);

        // Extra auto setups
        // Full viewport
        command_buffer.set_viewport_0(vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: size.width as f32,
            height: size.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        });
        // Full scissor
        command_buffer.set_scissor_0(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: size,
        });
    }

    fn end_graphics(&mut self, command_buffer: &CommandBuffer) {
        command_buffer.end_rendering();
    }

    fn bind_resources(
        &self,
        ctx: &RenderGraphExecuteContext,
        rd: &RenderDevice,
        pipeline: Option<&Pipeline>,
        command_buffer: &CommandBuffer,
        pass: &RenderPass,
        internal_set: vk::DescriptorSet,
    ) -> Option<()> {
        let pipeline_bind_point = pass.ty.to_bind_point()?;

        // Check for any resources
        let has_internal_set = internal_set != vk::DescriptorSet::null();
        let has_global_set = self.global_descriptor_sets.len() > 0;
        let any_resource = has_internal_set || has_global_set;
        if !any_resource {
            return None;
        }

        if pipeline.is_none() {
            println!(
                "Warning[RenderGraph::{}]: bind_resources failed: pipeline is none.",
                pass.name
            );
            return None;
        }
        let pipeline = pipeline.unwrap();

        // Update descriptor set
        // TODO this struct can be reused (a lot vec)
        if has_internal_set {
            let mut builder = shader::DescriptorSetWriteBuilder::new();

            let builder: &mut shader::DescriptorSetWriteBuilder<'_> = &mut builder;
            for (name, handle) in &pass.textures {
                let view = self.get_texture_view(ctx, *handle);
                builder.image(
                    name,
                    view.image_view,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
            }
            for (name, handle) in &pass.rw_textures {
                let view = self.get_texture_view(ctx, *handle);
                builder.image(name, view.image_view, vk::ImageLayout::GENERAL);
            }
            for (name, handle) in &pass.buffers {
                let buffer = self.get_buffer(ctx, *handle);
                builder.buffer(name, buffer.handle);
            }
            for (name, handle) in &pass.rw_buffers {
                let buffer = self.get_buffer(ctx, *handle);
                builder.buffer(name, buffer.handle);
            }
            for (name, handle) in &pass.rw_texel_buffers {
                let buffer_view = self.get_buffer_view(ctx, *handle);
                builder.texel_buffer(name, buffer_view.handle);
            }
            for (name, handle) in &pass.accel_structs {
                let accel_struct = self.get_accel_struct(*handle);
                builder.accel_struct(name, accel_struct.handle);
            }
            let writes = builder.build(
                pipeline,
                internal_set,
                pass.descriptor_set_index,
                |prop_name, ty_name| {
                    println!(
                        "Warning[RenderGraph::{}]: property {}:{} is not found.",
                        pass.name, prop_name, ty_name
                    );
                },
            );
            if !writes.is_empty() {
                unsafe {
                    rd.device.update_descriptor_sets(&writes, &[]);
                }
            }

            // Bind set
            command_buffer.bind_descriptor_set(
                pipeline_bind_point,
                pipeline.layout,
                pass.descriptor_set_index,
                internal_set,
                None,
            );
        }

        // Set globally bound sets
        // TODO should be be set evey pass? Can't it be set once per queue/command-buffer??
        for (set_index, set) in &self.global_descriptor_sets {
            if (*set_index == pass.descriptor_set_index)
                && (internal_set != vk::DescriptorSet::null())
            {
                println!("Error: RenderPass {} global set index {} is conflicted with internal set index {}", pass.name, set_index, pass.descriptor_set_index);
                continue;
            }

            if !pipeline.used_set.contains_key(set_index) {
                // Since global sets exist only for ergonomics purpose, it is totally okay if not used
                continue;
            }

            command_buffer.bind_descriptor_set(
                pipeline_bind_point,
                pipeline.layout,
                *set_index,
                *set,
                None,
            );
        }

        Some(())
    }

    fn ad_hoc_transition(
        &self,
        command_buffer: &CommandBuffer,
        ctx: &mut RenderGraphExecuteContext,
        pass_index: u32,
    ) {
        // helper
        let map_stage_mask = |pass: &RenderPass| match pass.ty {
            RenderPassType::Graphics(_) => vk::PipelineStageFlags::ALL_GRAPHICS, // TODO fragment access? vertex access? color output?
            RenderPassType::Compute(_) => vk::PipelineStageFlags::COMPUTE_SHADER,
            RenderPassType::RayTracing(_) => vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            RenderPassType::Present(_) => vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        };

        let get_last_access = |image: vk::Image,
                               aspect: vk::ImageAspectFlags,
                               end_pass_index: u32|
         -> Option<(u32, vk::AccessFlags, vk::ImageLayout)> {
            for pass_index in (0..end_pass_index).rev() {
                let pass = &self.passes[pass_index as usize];
                // Check all mutating view
                if let Some(gfx) = pass.ty.gfx() {
                    for rt in &gfx.color_targets {
                        let rt_view = self.get_texture_view(ctx, rt.view);
                        if (rt_view.texture.image == image) && (rt_view.desc.aspect == aspect) {
                            // TODO RenderPass Load ops?
                            let mut access = vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
                            if gfx.desc.blend_enabled {
                                access |= vk::AccessFlags::COLOR_ATTACHMENT_READ;
                            }
                            return Some((
                                pass_index,
                                access,
                                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                            ));
                        }
                    }
                    if let Some(ds) = &gfx.depth_stencil {
                        let view = self.get_texture_view(ctx, ds.view);
                        if (view.texture.image == image) && (view.desc.aspect == aspect) {
                            // TODO RenderPass Load ops?
                            let mut access = vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
                            // TODO READ only with depth/stencil test is enabled
                            access |= vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ;
                            return Some((
                                pass_index,
                                access,
                                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                            ));
                        }
                    }
                }
                for rw in &pass.rw_textures {
                    let rw_view = self.get_texture_view(ctx, rw.1);
                    if (rw_view.texture.image == image) && (rw_view.desc.aspect == aspect) {
                        // TODO render pass can specify write-only
                        let access = vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ;
                        return Some((pass_index, access, vk::ImageLayout::GENERAL));
                    }
                }
                // Check all sampling view
                for tex_view in &pass.textures {
                    let tex_view = self.get_texture_view(ctx, tex_view.1);
                    if (tex_view.texture.image == image) && (tex_view.desc.aspect == aspect) {
                        return Some((
                            pass_index,
                            vk::AccessFlags::SHADER_READ,
                            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        ));
                    }
                }
            }
            None
        };

        let pass = &self.passes[pass_index as usize];

        let transition_to = |image: vk::Image,
                             range: vk::ImageSubresourceRange,
                             dst_access_mask: vk::AccessFlags,
                             new_layout: vk::ImageLayout| {
            if let Some((last_pass_index, src_access_mask, last_layout)) =
                get_last_access(image, vk::ImageAspectFlags::COLOR, pass_index)
            {
                command_buffer.transition_image_layout(
                    image,
                    src_access_mask,
                    dst_access_mask,
                    map_stage_mask(&self.passes[last_pass_index as usize]),
                    map_stage_mask(pass),
                    last_layout,
                    new_layout,
                    range,
                )
            } else {
                // TODO support temporal resource
                command_buffer.transition_image_layout(
                    image,
                    vk::AccessFlags::empty(),
                    dst_access_mask,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    map_stage_mask(pass),
                    vk::ImageLayout::UNDEFINED,
                    new_layout,
                    range,
                )
            };
        };

        let transition_view_to = |handle: RGHandle<TextureView>,
                                  dst_access_mask: vk::AccessFlags,
                                  layout: vk::ImageLayout| {
            let view = self.get_texture_view(ctx, handle);
            transition_to(
                view.texture.image,
                view.desc.make_subresource_range(true),
                dst_access_mask,
                layout,
            );
        };

        for (_name, handle) in &pass.textures {
            transition_view_to(
                *handle,
                vk::AccessFlags::SHADER_READ,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
        }

        for (_name, handle) in &pass.rw_textures {
            let dst_access_mask = vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE;
            transition_view_to(*handle, dst_access_mask, vk::ImageLayout::GENERAL);
        }

        if let Some(gfx) = pass.ty.gfx() {
            for (_rt_index, rt) in gfx.color_targets.iter().enumerate() {
                let mut dst_access_mask = vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
                if gfx.desc.blend_enabled {
                    dst_access_mask |= vk::AccessFlags::COLOR_ATTACHMENT_READ;
                }
                transition_view_to(
                    rt.view,
                    dst_access_mask,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                );
            }

            if let Some(ds) = &gfx.depth_stencil {
                // TODO check depth test enabled
                let dst_access_mask = vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
                transition_view_to(
                    ds.view,
                    dst_access_mask,
                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL, // TODO finer grain for depth-stencil access
                );
            }
        }

        // Special: check present, transition after last access
        if let Some(present) = pass.ty.present() {
            let present_tex = self.get_texture(ctx, present.texture);
            transition_to(
                present_tex.image,
                vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                vk::AccessFlags::NONE, // seems suggest by Vulkan-Doc
                vk::ImageLayout::PRESENT_SRC_KHR,
            );
        }

        // Buffers

        let get_last_access =
            |vk_buffer: vk::Buffer, end_pass_index: u32| -> Option<(u32, vk::AccessFlags)> {
                for pass_index in (0..end_pass_index).rev() {
                    let pass = &self.passes[pass_index as usize];

                    // Check all mutating
                    for (_name, handle) in &pass.rw_buffers {
                        let buffer = self.get_buffer(ctx, *handle);
                        if buffer.handle == vk_buffer {
                            return Some((pass_index, vk::AccessFlags::SHADER_WRITE));
                        }
                    }
                    for (_name, handle) in &pass.rw_texel_buffers {
                        let texel_buffer = self.get_buffer_view(ctx, *handle);
                        if texel_buffer.buffer.handle == vk_buffer {
                            return Some((pass_index, vk::AccessFlags::SHADER_WRITE));
                        }
                    }

                    // Check all sampling
                    for (_name, handle) in &pass.buffers {
                        let buffer = self.get_buffer(ctx, *handle);
                        if buffer.handle == vk_buffer {
                            return Some((pass_index, vk::AccessFlags::SHADER_READ));
                        }
                    }
                }
                None
            };

        let transition_to = |vk_buffer: vk::Buffer, dst_access_mask: vk::AccessFlags| {
            if let Some((last_pass_index, src_access_mask)) = get_last_access(vk_buffer, pass_index)
            {
                if dst_access_mask != src_access_mask {
                    let barrier = vk::BufferMemoryBarrier::builder()
                        .src_access_mask(src_access_mask)
                        .dst_access_mask(dst_access_mask)
                        //.src_queue_family_index(0) // ?
                        //.dst_queue_family_index(0) // ?
                        .buffer(vk_buffer)
                        .offset(0)
                        .size(vk::WHOLE_SIZE);

                    command_buffer.pipeline_barrier(
                        map_stage_mask(&self.passes[last_pass_index as usize]),
                        map_stage_mask(pass),
                        vk::DependencyFlags::empty(), // TODO do it properly
                        &[],
                        std::slice::from_ref(&barrier),
                        &[],
                    );
                }
            } else {
                // Nothing to sync
                // TODO support temporal resource
            };
        };

        for (_name, handle) in &pass.buffers {
            transition_to(
                self.get_buffer(ctx, *handle).handle,
                vk::AccessFlags::SHADER_READ,
            );
        }

        for (_name, handle) in &pass.rw_buffers {
            transition_to(
                self.get_buffer(ctx, *handle).handle,
                vk::AccessFlags::SHADER_WRITE,
            );
        }

        for (_name, handle) in &pass.rw_texel_buffers {
            transition_to(
                self.get_buffer_view(ctx, *handle).buffer.handle,
                vk::AccessFlags::SHADER_WRITE,
            )
        }

        for (_name, handle) in &pass.accel_structs {
            transition_to(
                self.get_accel_struct(*handle).buffer.handle,
                vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
            );
        }
    }

    pub fn done(self) -> RenderGraphCache {
        self.cache
    }

    pub fn execute(
        &mut self,
        rd: &RenderDevice,
        command_buffer: &CommandBuffer,
        shaders: &mut Shaders,
    ) {
        puffin::profile_function!();

        let mut exec_context = RenderGraphExecuteContext::new();

        command_buffer.insert_label(&CString::new("RenderGraph Begin").unwrap(), None);

        // Update GPU profiling queries
        self.cache.pass_profiling.update(rd);

        // Prepare for this frame
        let query_pool = {
            let batch = self
                .cache
                .pass_profiling
                .new_batch(self.passes.len() as u32 + 1, self.hack_frame_index);
            command_buffer.reset_queries(batch.pool(), batch.query(0).0, batch.size());
            batch.pool()
        };

        // Whole Frame timer
        let (frame_beg, frame_end) = self.cache.pass_profiling.new_timer("[Frame]");
        command_buffer.write_time_stamp(
            vk::PipelineStageFlags::TOP_OF_PIPE,
            query_pool,
            frame_beg.0,
        );

        // Create pipelines
        for pass_index in 0..self.passes.len() {
            let pass = &mut self.passes[pass_index];

            let pipeline = match &pass.ty {
                RenderPassType::Graphics(gfx) => {
                    // very hack
                    let mut hack = ShadersConfig {
                        set_layout_override: self.shader_config.set_layout_override.clone(),
                    };
                    if let Some(o) = &pass.set_layout_override {
                        hack.set_layout_override.insert(o.0, o.1);
                    }
                    shaders
                        .create_gfx_pipeline(
                            gfx.vertex_shader.unwrap(),
                            gfx.pixel_shader.unwrap(),
                            &gfx.desc,
                            &hack,
                        )
                        .unwrap()
                }
                RenderPassType::Compute(compute) => shaders
                    .create_compute_pipeline(compute.shader.unwrap(), &self.shader_config)
                    .unwrap(),
                RenderPassType::RayTracing(rt) => shaders
                    .create_raytracing_pipeline(
                        rt.raygen_shader.unwrap(),
                        &rt.miss_shaders,
                        rt.chit_shader,
                        &rt.desc,
                        &self.shader_config,
                    )
                    .unwrap(),
                _ => Handle::<Pipeline>::null(),
            };

            exec_context.pipelines.push(pipeline);
        }

        // Populate textures and views
        // TODO drain self.texture, self.texture_views to context
        // TODO memory aliasing
        for i in 0..self.textures.len() {
            let texture_resource = &self.textures[i];
            let texture = match texture_resource {
                // Create texture
                RenderResource::Virtual(desc) => {
                    let tex = self
                        .cache
                        .texture_pool
                        .pop(&desc)
                        .unwrap_or_else(|| rd.create_texture(*desc).unwrap());
                    Some(tex)
                }
                RenderResource::Temporal(temporal_id) => {
                    let tex = self.cache.temporal_textures.remove(temporal_id).unwrap();
                    Some(tex)
                }
                RenderResource::External(_) => None,
            };
            exec_context.textures.push(texture);
        }
        for i in 0..self.texture_views.len() {
            let view_resource = &self.texture_views[i];
            let view = match view_resource {
                // Create texture view
                RenderResource::Virtual(virtual_view) => {
                    let texture = self.get_texture(&exec_context, virtual_view.texture);
                    let view_desc = virtual_view
                        .desc
                        .unwrap_or_else(|| TextureViewDesc::auto(&texture.desc));
                    let view = self
                        .cache
                        .texture_view_pool
                        .pop(&(texture, view_desc))
                        .unwrap_or_else(|| rd.create_texture_view(texture, view_desc).unwrap());
                    Some(view)
                }
                RenderResource::Temporal(_) => panic!("No temporal texture view"),
                RenderResource::External(_) => None,
            };
            exec_context.texture_views.push(view);
        }
        for i in 0..self.buffers.len() {
            let res = &self.buffers[i];
            let buffer = match res {
                RenderResource::Virtual(desc) => {
                    let buf = self
                        .cache
                        .buffer_pool
                        .pop(&desc)
                        .unwrap_or_else(|| rd.create_buffer(*desc).unwrap());
                    Some(buf)
                }
                RenderResource::Temporal(temporal_id) => {
                    let buf = self.cache.temporal_buffers.remove(temporal_id).unwrap();
                    Some(buf)
                }
                RenderResource::External(_) => None,
            };
            exec_context.buffers.push(buffer);
        }
        for i in 0..self.buffer_views.len() {
            let res = &self.buffer_views[i];
            let view = match res {
                // Create
                RenderResource::Virtual(virtual_view) => {
                    let buffer = self.get_buffer(&exec_context, virtual_view.buffer);
                    let desc = virtual_view.desc;
                    let view = self
                        .cache
                        .buffer_view_pool
                        .pop(&(buffer, desc))
                        .unwrap_or_else(|| rd.create_buffer_view(buffer, desc).unwrap());
                    Some(view)
                }
                RenderResource::Temporal(_) => panic!("No temporal buffer view"),
                RenderResource::External(_) => None,
            };
            exec_context.buffer_views.push(view);
        }

        // Create (temp) descriptor set for each pass
        for (pass_index, pass) in self.passes.iter().enumerate() {
            let pipeline = shaders.get_pipeline(exec_context.pipelines[pass_index]);
            let set = match pipeline {
                Some(pipeline) => {
                    if pass.descriptor_set_index as usize >= pipeline.set_layouts.len() {
                        /* Totally happen in shader debug
                        println!(
                            "Warning: unused descriptor set {} for pass {}",
                            pass.descriptor_set_index, pass.name
                        );
                        */
                        vk::DescriptorSet::null()
                    } else {
                        let set_layout = pipeline.set_layouts[pass.descriptor_set_index as usize];
                        self.cache.allocate_dessriptor_set(rd, set_layout)
                    }
                }
                None => {
                    if !pass.ty.is_present() {
                        println!("Warning[RenderGraph]: pipeline not provided by pass {}; temporal descriptor set is not created.", pass.name);
                    }
                    vk::DescriptorSet::null()
                }
            };

            exec_context.descriptor_sets.push(set);
        }
        assert!(exec_context.descriptor_sets.len() == self.passes.len());

        // [raytracing]
        // Create and update ShaderBindingTable for each ray tracing pass
        let sbt_frame_index = self.hack_frame_index % 3;
        for (pass_index, pass) in self.passes.iter().enumerate() {
            let pass_sbt = match &pass.ty {
                RenderPassType::RayTracing(rt) => {
                    let mut sbt = self
                        .cache
                        .sbt_pool
                        .pop(&sbt_frame_index) // hack
                        .unwrap_or_else(|| PassShaderBindingTable::new(rd));

                    // update anyway :)
                    // TODO using frame index % 3 to void cpu-write-on-GPU-read; should do it with proper synchronization
                    let num_miss = rt.miss_shaders.len() as u32;
                    let has_hit = rt.chit_shader.is_some();
                    let pipeline = shaders
                        .get_pipeline(exec_context.pipelines[pass_index])
                        .unwrap();
                    sbt.update_shader_group_handles(rd, pipeline, num_miss, has_hit);

                    Some(sbt)
                }
                _ => None,
            };
            exec_context.shader_binding_tables.push(pass_sbt);
        }

        for pass_index in 0..self.passes.len() {
            // take the FnOnce callback before unmutable reference
            let pre_render = self.passes[pass_index].pre_render.take();
            let render = self.passes[pass_index].render.take();

            let pass = &self.passes[pass_index];

            command_buffer.begin_label(&CString::new(pass.name.clone()).unwrap(), None);

            let pipeline = shaders.get_pipeline(exec_context.pipelines[pass_index]);

            // Some custom pre-render callback, to do external synchronization, etc.
            if let Some(pre_render) = pre_render {
                if let Some(pipeline) = pipeline {
                    pre_render(command_buffer, pipeline);
                } else {
                    println!("Error[RenderGraph]: pre_render callback is ignored because not valid pipeline.");
                }
            }

            // TODO analysis of the DAG and sync properly
            self.ad_hoc_transition(command_buffer, &mut exec_context, pass_index as u32);

            // Begin render pass (if graphics)
            let is_graphic = pass.ty.is_graphics();
            if is_graphic {
                self.begin_graphics(&exec_context, command_buffer, pass);
            }

            // Bind pipeline (if set)
            if let Some(pipeline) = pipeline {
                let pipeline_bind_point = pass.ty.to_bind_point().unwrap();
                command_buffer.bind_pipeline(pipeline_bind_point, pipeline.handle);
            }

            // Bind resources
            self.bind_resources(
                &exec_context,
                rd,
                pipeline,
                command_buffer,
                &pass,
                exec_context.descriptor_sets[pass_index],
            );

            // Push Constant (if pushed)
            let pc_data = pass.push_constants.build();
            if let Some(pipeline) = pipeline {
                if (pc_data.len() > 0) && (pipeline.push_constant_ranges.len() > 0) {
                    let range = pipeline.push_constant_ranges[0];
                    command_buffer.push_constants(pipeline.layout, range.stage_flags, 0, pc_data);
                } else if pc_data.len() > 0 {
                    println!(
                        "Warning[RenderGraph]: push constant is not used by pass {}",
                        pass.name
                    );
                } else if pipeline.push_constant_ranges.len() > 0 {
                    // It is okay the render pass set push constant range them self in custom callback
                    if render.is_none() {
                        println!(
                            "Warning[RenderGraph]: push constant (or custom render callback) is not provided for pass {}",
                            pass.name
                        );
                    }
                }
            } else if pc_data.len() > 0 {
                println!(
                    "Warning[RenderGraph]: pipeline is not provided for pass {}",
                    pass.name
                );
            }

            // Timestamp: before executing the work
            let (timer_beg, timer_end) = self.cache.pass_profiling.new_timer(&pass.name);
            command_buffer.write_time_stamp(
                vk::PipelineStageFlags::TOP_OF_PIPE,
                query_pool,
                timer_beg.0,
            );

            // Run pass
            // prioritize the custom render function
            if let Some(render) = render {
                if let Some(pipeline) = pipeline {
                    render(command_buffer, pipeline);
                } else {
                    println!("Error[RenderGraph]: render callback is ignored because not valid pipeline.");
                }
            }
            // automatically call the thing
            else {
                match &pass.ty {
                    RenderPassType::Graphics(_) => {
                        println!(
                            "Error[RenderGraph]: render callback is not provided to graphics pass."
                        );
                    }
                    RenderPassType::Compute(compute) => {
                        let group_count = &compute.group_count;

                        let max_group_count = UVec3::from_array(
                            rd.physical.properties.limits.max_compute_work_group_count,
                        );
                        if (group_count.x > max_group_count.x)
                            || (group_count.y > max_group_count.y)
                            || (group_count.z > max_group_count.z)
                        {
                            print!("Error[RenderGraph]: compute group count exceed hardware limit {} > {}", group_count, max_group_count);
                        }

                        command_buffer.dispatch(group_count.x, group_count.y, group_count.z);
                    }
                    RenderPassType::RayTracing(rt) => {
                        let has_hit = rt.chit_shader.is_some();
                        let sbt = exec_context.shader_binding_tables[pass_index]
                            .as_ref()
                            .unwrap();

                        let empty_region = vk::StridedDeviceAddressRegionKHR::default();
                        let hit_region = if has_hit {
                            &sbt.hit_region
                        } else {
                            &empty_region
                        };

                        let dim = rt.dimension;
                        command_buffer.trace_rays(
                            &sbt.raygen_region,
                            &sbt.miss_region,
                            hit_region,
                            &empty_region,
                            dim.x,
                            dim.y,
                            dim.z,
                        );
                    }
                    RenderPassType::Present(_) => {
                        // nothing to call
                    }
                }
            }

            // End render pass (if graphics)
            if is_graphic {
                self.end_graphics(command_buffer);
            }

            // Timestamp: after executing the work
            command_buffer.write_time_stamp(
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                query_pool,
                timer_end.0,
            );

            command_buffer.end_label();
        }

        // Process all textures converted to temporals
        for (handle, temporal_handle) in &self.transient_to_temporal_textures {
            let tex = exec_context.textures[handle.id].take().unwrap();
            let conflict = self.cache.temporal_textures.insert(temporal_handle.id, tex);
            assert!(conflict.is_none());
        }
        self.transient_to_temporal_textures.clear();
        for (handle, temporal_handle) in &self.transient_to_temporal_buffers {
            let buf = exec_context.buffers[handle.id].take().unwrap();
            let conflict = self.cache.temporal_buffers.insert(temporal_handle.id, buf);
            assert!(conflict.is_none());
        }

        // puch back temporal id allocator
        //assert!(cache.next_temporal_id.is_none());
        //cache.next_temporal_id = Some(self.next_temporal_id);

        // Pool back all resource objects
        for view in exec_context
            .texture_views
            .drain(0..exec_context.texture_views.len())
        {
            if let Some(view) = view {
                self.cache
                    .texture_view_pool
                    .push((view.texture, view.desc), view);
            }
        }
        for texture in exec_context.textures.drain(0..exec_context.textures.len()) {
            if let Some(texture) = texture {
                self.cache.texture_pool.push(texture.desc, texture);
            }
        }
        for buffer in exec_context.buffers.drain(0..exec_context.buffers.len()) {
            if let Some(buffer) = buffer {
                self.cache.buffer_pool.push(buffer.desc, buffer);
            }
        }
        for view in exec_context
            .buffer_views
            .drain(0..exec_context.buffer_views.len())
        {
            if let Some(view) = view {
                self.cache
                    .buffer_view_pool
                    .push((view.buffer, view.desc), view);
            }
        }
        for set in exec_context
            .descriptor_sets
            .drain(0..exec_context.descriptor_sets.len())
        {
            if set != vk::DescriptorSet::null() {
                // TODO may be need some frame buffering, because it may be still using?
                self.cache.release_descriptor_set(rd, set);
            }
        }
        for sbt in exec_context
            .shader_binding_tables
            .drain(0..exec_context.shader_binding_tables.len())
        {
            if let Some(sbt) = sbt {
                self.cache.sbt_pool.push(sbt_frame_index, sbt);
            }
        }

        command_buffer.write_time_stamp(
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            query_pool,
            frame_end.0,
        );

        command_buffer.insert_label(&CString::new("RenderGraph End").unwrap(), None);
    }
}
