use ash::vk::{self};

pub struct CommandBuffer {
    pub device: ash::Device,
    pub command_buffer: vk::CommandBuffer,
}

pub struct StencilOps {
    fail_op: vk::StencilOp,
    pass_op: vk::StencilOp,
    depth_fail_op: vk::StencilOp,
    compare_op: vk::CompareOp,
}

impl StencilOps {
    pub fn write_on_pass(compare_op: vk::CompareOp) -> Self {
        Self {
            fail_op: vk::StencilOp::KEEP,
            pass_op: vk::StencilOp::REPLACE,
            depth_fail_op: vk::StencilOp::KEEP,
            compare_op,
        }
    }

    pub fn only_compare(compare_op: vk::CompareOp) -> Self {
        Self {
            fail_op: vk::StencilOp::KEEP,
            pass_op: vk::StencilOp::KEEP,
            depth_fail_op: vk::StencilOp::KEEP,
            compare_op,
        }
    }
}

impl CommandBuffer {
    pub fn transition_image_layout(
        &self,
        image: vk::Image,
        src_stage_mask: vk::PipelineStageFlags,
        dst_stage_mask: vk::PipelineStageFlags,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        subresource_range: vk::ImageSubresourceRange,
    ) {
        let image_barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .subresource_range(subresource_range)
            .image(image);
        self.pipeline_barrier(
            src_stage_mask,
            dst_stage_mask,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[*image_barrier],
        );
    }

    pub fn pipeline_barrier(
        &self,
        src_stage_mask: vk::PipelineStageFlags,
        dst_stage_mask: vk::PipelineStageFlags,
        dependency_flags: vk::DependencyFlags,
        memory_barriers: &[vk::MemoryBarrier],
        buffer_memory_barriers: &[vk::BufferMemoryBarrier],
        image_memory_barriers: &[vk::ImageMemoryBarrier],
    ) {
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                src_stage_mask,
                dst_stage_mask,
                dependency_flags,
                memory_barriers,
                buffer_memory_barriers,
                image_memory_barriers,
            );
        }
    }

    pub fn dispatch(&self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        unsafe {
            self.device.cmd_dispatch(
                self.command_buffer,
                group_count_x,
                group_count_y,
                group_count_z,
            );
        }
    }

    pub fn begin_rendering(&self, rendering_info: &vk::RenderingInfo) {
        unsafe {
            self.device
                .cmd_begin_rendering(self.command_buffer, rendering_info);
        }
    }

    pub fn end_rendering(&self) {
        unsafe {
            self.device.cmd_end_rendering(self.command_buffer);
        }
    }

    pub fn set_viewport_0(&self, viewport: vk::Viewport) {
        unsafe {
            self.device
                .cmd_set_viewport(self.command_buffer, 0, std::slice::from_ref(&viewport));
        }
    }

    pub fn set_scissor_0(&self, scissor: vk::Rect2D) {
        unsafe {
            self.device
                .cmd_set_scissor(self.command_buffer, 0, std::slice::from_ref(&scissor));
        }
    }

    pub fn set_depth_test_enable(&self, value: bool) {
        unsafe {
            self.device
                .cmd_set_depth_test_enable(self.command_buffer, value);
        }
    }

    pub fn set_depth_write_enable(&self, value: bool) {
        unsafe {
            self.device
                .cmd_set_depth_write_enable(self.command_buffer, value);
        }
    }

    pub fn set_stencil_test_enable(&self, value: bool) {
        unsafe {
            self.device
                .cmd_set_stencil_test_enable(self.command_buffer, value);
        }
    }

    pub fn set_stencil_op(&self, face_mask: vk::StencilFaceFlags, ops: StencilOps) {
        unsafe {
            self.device.cmd_set_stencil_op(
                self.command_buffer,
                face_mask,
                ops.fail_op,
                ops.pass_op,
                ops.depth_fail_op,
                ops.compare_op,
            )
        }
    }

    pub fn set_stencil_write_mask(&self, face_mask: vk::StencilFaceFlags, write_mask: u32) {
        unsafe {
            self.device
                .cmd_set_stencil_write_mask(self.command_buffer, face_mask, write_mask)
        }
    }

    pub fn set_stencil_compare_mask(&self, face_mask: vk::StencilFaceFlags, compare_mask: u32) {
        unsafe {
            self.device
                .cmd_set_stencil_compare_mask(self.command_buffer, face_mask, compare_mask)
        }
    }

    pub fn set_stencil_reference(&self, face_mask: vk::StencilFaceFlags, reference: u32) {
        unsafe {
            self.device
                .cmd_set_stencil_reference(self.command_buffer, face_mask, reference);
        }
    }

    pub fn bind_descriptor_set(
        &self,
        pipeline_bind_point: vk::PipelineBindPoint,
        pipeline_layout: vk::PipelineLayout,
        set_index: u32,
        set: vk::DescriptorSet,
        dynamic_offset: Option<u32>,
    ) {
        unsafe {
            let dynamic_offset_count = if dynamic_offset.is_some() { 1 } else { 0 };
            let dynamic_offset = dynamic_offset.unwrap_or(0);
            self.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                pipeline_bind_point,
                pipeline_layout,
                set_index,
                std::slice::from_ref(&set),
                std::slice::from_raw_parts(&dynamic_offset, dynamic_offset_count),
            );
        }
    }

    pub fn bind_index_buffer(&self, buffer: vk::Buffer, offset: u64, index_type: vk::IndexType) {
        unsafe {
            self.device
                .cmd_bind_index_buffer(self.command_buffer, buffer, offset, index_type);
        }
    }

    pub fn bind_pipeline(
        &self,
        pipeline_bind_point: vk::PipelineBindPoint,
        pipeline: vk::Pipeline,
    ) {
        unsafe {
            self.device
                .cmd_bind_pipeline(self.command_buffer, pipeline_bind_point, pipeline);
        }
    }

    pub fn push_constants(
        &self,
        pipeline_layout: vk::PipelineLayout,
        stage_flags: vk::ShaderStageFlags,
        offset: u32,
        constants: &[u8],
    ) {
        unsafe {
            self.device.cmd_push_constants(
                self.command_buffer,
                pipeline_layout,
                stage_flags,
                offset,
                constants,
            )
        }
    }

    pub fn draw(
        &self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        unsafe {
            self.device.cmd_draw(
                self.command_buffer,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            );
        }
    }

    pub fn draw_indexed(
        &self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        unsafe {
            self.device.cmd_draw_indexed(
                self.command_buffer,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            )
        }
    }
}
