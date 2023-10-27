use ash::vk;

pub struct TextureUsage {
    vk: vk::ImageUsageFlags,
}

impl TextureUsage {
    pub fn from_vk(flags: vk::ImageUsageFlags) -> TextureUsage {
        TextureUsage { vk: flags }
    }

    pub fn compute() -> TextureUsage {
        Self::from_vk(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
    }

    pub fn to_vk(self) -> vk::ImageUsageFlags {
        self.vk
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct TextureDesc {
    pub width: u32,
    pub height: u32,
    pub layer_count: u32,
    pub mip_level_count: u32,
    pub format: vk::Format,
    pub usage: vk::ImageUsageFlags,
    pub flags: vk::ImageCreateFlags,
}

impl Default for TextureDesc {
    fn default() -> Self {
        Self {
            width: 1,
            height: 1,
            layer_count: 1,
            mip_level_count: 1,
            format: vk::Format::UNDEFINED,
            usage: vk::ImageUsageFlags::default(),
            flags: vk::ImageCreateFlags::default(),
        }
    }
}

impl TextureDesc {
    pub fn new_2d(
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> TextureDesc {
        TextureDesc {
            width,
            height,
            layer_count: 1,
            mip_level_count: 1,
            format,
            usage,
            flags: vk::ImageCreateFlags::default(),
        }
    }

    pub fn new_2d_array(
        width: u32,
        height: u32,
        array_len: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> TextureDesc {
        TextureDesc {
            width,
            height,
            layer_count: array_len,
            mip_level_count: 1,
            format,
            usage,
            flags: vk::ImageCreateFlags::default(),
        }
    }

    pub fn with_flags(mut self, flag: vk::ImageCreateFlags) -> Self {
        self.flags = flag;
        self
    }

    pub fn size_2d(&self) -> vk::Extent2D {
        vk::Extent2D {
            width: self.width,
            height: self.height,
        }
    }

    pub fn size_3d(&self) -> vk::Extent3D {
        vk::Extent3D {
            width: self.width,
            height: self.height,
            depth: self.layer_count,
        }
    }
}

// Mini struct for a texture
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct Texture {
    pub desc: TextureDesc,
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
}

/*
impl PartialEq for Texture {
    fn eq(&self, other: &Self) -> bool {
        self.image == other.image
    }
}

impl Eq for Texture {}
 */

// Mini struct for a texture view
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct TextureViewDesc {
    pub view_type: vk::ImageViewType,
    pub format: vk::Format,
    pub aspect: vk::ImageAspectFlags,
    pub base_mip_level: u32,
    pub level_count: u32,
    pub base_array_layer: u32,
    pub layer_count: u32,
}

impl Default for TextureViewDesc {
    fn default() -> Self {
        TextureViewDesc {
            view_type: vk::ImageViewType::TYPE_2D,
            format: vk::Format::UNDEFINED,
            aspect: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: vk::REMAINING_MIP_LEVELS,
            base_array_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
        }
    }
}

pub fn format_has_depth(format: vk::Format) -> bool {
    (format == vk::Format::D16_UNORM)
        || (format == vk::Format::D16_UNORM_S8_UINT)
        || (format == vk::Format::D24_UNORM_S8_UINT)
        || (format == vk::Format::D32_SFLOAT)
        || (format == vk::Format::D32_SFLOAT_S8_UINT)
}

pub fn format_has_stencil(format: vk::Format) -> bool {
    (format == vk::Format::S8_UINT)
        || (format == vk::Format::D16_UNORM_S8_UINT)
        || (format == vk::Format::D24_UNORM_S8_UINT)
        || (format == vk::Format::D32_SFLOAT_S8_UINT)
}

pub fn format_has_depth_stencil(format: vk::Format) -> bool {
    (format == vk::Format::D16_UNORM_S8_UINT)
        || (format == vk::Format::D24_UNORM_S8_UINT)
        || (format == vk::Format::D32_SFLOAT_S8_UINT)
}

impl TextureViewDesc {
    pub fn auto(texture_desc: &TextureDesc) -> TextureViewDesc {
        let view_type = if texture_desc.layer_count > 1 {
            vk::ImageViewType::TYPE_2D_ARRAY
        } else {
            vk::ImageViewType::TYPE_2D
        };
        let format = texture_desc.format;
        let has_depth = format_has_depth(format);
        let has_stencil = format_has_stencil(format);
        let aspect = if has_depth {
            vk::ImageAspectFlags::DEPTH
        } else {
            if has_stencil {
                vk::ImageAspectFlags::STENCIL
            } else {
                vk::ImageAspectFlags::COLOR
            }
        };
        TextureViewDesc {
            view_type,
            format: texture_desc.format,
            aspect,
            base_mip_level: 0,
            level_count: texture_desc.mip_level_count,
            base_array_layer: 0,
            layer_count: texture_desc.layer_count,
        }
    }

    pub fn with_format(texture_desc: &TextureDesc, format: vk::Format) -> TextureViewDesc {
        let mut desc = Self::auto(texture_desc);
        desc.format = format;
        desc
    }

    pub fn make_subresource_range(&self, for_transition: bool) -> vk::ImageSubresourceRange {
        /* Vulkan Spec: If image has a depth/stencil format with both depth and stencil and the separateDepthStencilLayouts feature is not enabled, then the aspectMask member of subresourceRange must include both VK_IMAGE_ASPECT_DEPTH_BIT and VK_IMAGE_ASPECT_STENCIL_BIT
         */
        let aspect = if for_transition && format_has_depth_stencil(self.format) {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        } else {
            self.aspect
        };
        assert!(aspect & self.aspect == self.aspect);
        vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: self.base_mip_level,
            level_count: self.level_count,
            base_array_layer: self.base_array_layer,
            layer_count: self.layer_count,
        }
    }

    pub fn make_subresrouce_layer(&self) -> vk::ImageSubresourceLayers {
        vk::ImageSubresourceLayers {
            aspect_mask: self.aspect,
            mip_level: self.base_mip_level,
            base_array_layer: self.base_array_layer,
            layer_count: self.layer_count,
        }
    }
}

#[derive(Clone, Copy)]
pub struct TextureView {
    pub texture: Texture,
    pub desc: TextureViewDesc,
    pub image_view: vk::ImageView,
}

impl PartialEq for TextureView {
    fn eq(&self, other: &Self) -> bool {
        self.image_view == other.image_view
    }
}

// NOTE: not perfect match! (e.g. atomic stoge, color blend)
fn image_usage_to_feature(usage: vk::ImageUsageFlags) -> vk::FormatFeatureFlags {
    let mut features = vk::FormatFeatureFlags::empty();
    use vk::FormatFeatureFlags as Feature;
    use vk::ImageUsageFlags as Usage;
    let pairs = [
        (Usage::SAMPLED, Feature::SAMPLED_IMAGE),
        (Usage::STORAGE, Feature::STORAGE_IMAGE),
        (Usage::COLOR_ATTACHMENT, Feature::COLOR_ATTACHMENT),
        (
            Usage::DEPTH_STENCIL_ATTACHMENT,
            Feature::DEPTH_STENCIL_ATTACHMENT,
        ),
    ];
    for p in pairs {
        if usage.contains(p.0) {
            features = features | p.1;
        }
    }
    features
}

impl super::RenderDevice {
    pub fn create_texture(&self, desc: TextureDesc) -> Option<Texture> {
        let format_prop = unsafe {
            self.instance.get_physical_device_image_format_properties(
                self.physical.handle,
                desc.format,
                vk::ImageType::TYPE_2D,
                vk::ImageTiling::OPTIMAL,
                desc.usage,
                vk::ImageCreateFlags::default(),
            )
        };
        if let Err(e) = format_prop {
            println!(
                "Error: texture creation for {:?} failed: {:?}. Try something else.",
                desc.format, e
            );
            match e {
                vk::Result::ERROR_FORMAT_NOT_SUPPORTED => {
                    let prop = self.physical.get_format_properties(desc.format);
                    println!("Format not supported. Format properties: {:?}", prop);

                    // Hint log
                    let features = image_usage_to_feature(desc.usage);
                    let tilings = [vk::ImageTiling::OPTIMAL, vk::ImageTiling::LINEAR];
                    for tiling in tilings {
                        println!(
                            "Try these formats for usage {:?} with {:?} tiling:",
                            desc.usage, tiling
                        );
                        let supported_formats =
                            self.physical.list_supported_image_formats(tiling, features);
                        for format in supported_formats {
                            println!("\t{:?}", format);
                        }
                    }
                }
                _ => {}
            }
            return None;
        }

        // Create image object
        let initial_layout = vk::ImageLayout::UNDEFINED;
        let create_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(desc.format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .extent(vk::Extent3D {
                width: desc.width,
                height: desc.height,
                depth: 1,
            })
            .array_layers(desc.layer_count)
            .mip_levels(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .initial_layout(initial_layout)
            .usage(desc.usage)
            .flags(desc.flags);
        let image = unsafe { self.device.create_image(&create_info, None) }.unwrap();

        // Bind memory
        let device_memory = {
            let mem_requirements = unsafe { self.device.get_image_memory_requirements(image) };
            let momory_type_index = self.physical.pick_memory_type_index(
                mem_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL, // That's basically what texture can have
            )?;
            let create_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(mem_requirements.size)
                .memory_type_index(momory_type_index);
            unsafe { self.device.allocate_memory(&create_info, None) }.unwrap()
        };
        unsafe { self.device.bind_image_memory(image, device_memory, 0) }.unwrap();

        Some(Texture {
            desc,
            image,
            memory: device_memory,
        })
    }

    pub fn create_texture_view(
        &self,
        texture: Texture,
        desc: TextureViewDesc,
    ) -> Option<TextureView> {
        let create_info = vk::ImageViewCreateInfo::builder()
            .image(texture.image)
            .view_type(desc.view_type)
            .format(desc.format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: desc.aspect,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: texture.desc.layer_count,
            });
        let image_view = unsafe { self.device.create_image_view(&create_info, None) }.ok()?;

        Some(TextureView {
            texture,
            desc,
            image_view,
        })
    }
}
