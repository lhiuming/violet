use ash::vk;

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct BufferDesc {
    pub size: u64,
    pub usage: vk::BufferUsageFlags,
    pub memory_property: vk::MemoryPropertyFlags,
}

impl Default for BufferDesc {
    fn default() -> Self {
        Self {
            size: 0,
            usage: vk::BufferUsageFlags::empty(),
            memory_property: vk::MemoryPropertyFlags::empty(),
        }
    }
}

impl BufferDesc {
    // Read/Write in GPU
    pub fn compute(size: u64) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_property: vk::MemoryPropertyFlags::DEVICE_LOCAL,
        }
    }

    // Write (typically once) in CPU, read in GPU
    pub fn shader_binding_table(size: u64) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            memory_property: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT, // TODO no need to coherent?
        }
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct Buffer {
    pub desc: BufferDesc,
    pub handle: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub data: *mut u8, // TODO make optional
    pub device_address: Option<vk::DeviceAddress>,
}

impl super::RenderDevice {
    pub fn create_buffer(&self, desc: BufferDesc) -> Option<Buffer> {
        // Create the vk buffer object
        // TODO drop buffer if later stage failed
        let buffer = {
            let create_info = vk::BufferCreateInfo::builder()
                .size(desc.size)
                .usage(desc.usage);
            match unsafe { self.device.create_buffer(&create_info, None) } {
                Ok(buffer) => buffer,
                Err(err) => {
                    println!("RenderDevice: (Vulkan) failed to create buffer: {:?}", err);
                    return None;
                }
            }
        };

        let has_device_address = desc
            .usage
            .contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);

        // Allocate memory for ths buffer
        // TODO drop device_memory if later stage failed
        // TODO use a allocator like VMA to do sub allocate
        let memory: vk::DeviceMemory = {
            let mem_req = unsafe { self.device.get_buffer_memory_requirements(buffer) };

            // Pick memory type
            /*
            let mem_property_flags =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            */
            let memory_type_index = self
                .physical
                .pick_memory_type_index(mem_req.memory_type_bits, desc.memory_property)
                .unwrap();

            let mut flags = vk::MemoryAllocateFlags::default();
            if has_device_address {
                // allocation requirement (03339)
                flags |= vk::MemoryAllocateFlags::DEVICE_ADDRESS;
            }
            let mut flag_info = vk::MemoryAllocateFlagsInfo::builder().flags(flags).build();

            let create_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(mem_req.size)
                .memory_type_index(memory_type_index)
                .push_next(&mut flag_info);
            match unsafe { self.device.allocate_memory(&create_info, None) } {
                Ok(mem) => mem,
                Err(err) => {
                    println!("RenderDevice: (Vulkan) failed to bind buffer: {:?}", err);
                    return None;
                }
            }
        };

        // Bind
        let offset: vk::DeviceSize = 0;
        match unsafe { self.device.bind_buffer_memory(buffer, memory, offset) } {
            Ok(_) => {}
            Err(err) => {
                println!("RenderDevice: (Vulkan) failed to bind buffer: {:?}", err);
                return None;
            }
        }

        // Get address (for later use, e.g. ray tracing)
        let device_address = if has_device_address {
            unsafe {
                let info = vk::BufferDeviceAddressInfo::builder().buffer(buffer);
                Some(self.device.get_buffer_device_address(&info))
            }
        } else {
            None
        };

        // Map (staging buffer) persistently
        // TODO unmap if later stage failed
        let is_mappable = desc
            .memory_property
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE);
        let data = if is_mappable {
            let map_flags = vk::MemoryMapFlags::default(); // dummy parameter
            unsafe { self.device.map_memory(memory, offset, desc.size, map_flags) }.unwrap()
                as *mut u8
        } else {
            std::ptr::null_mut::<u8>()
        };

        Some(Buffer {
            desc,
            handle: buffer,
            memory,
            data,
            device_address,
        })
    }

    pub fn create_buffer_view(
        &self,
        buffer: vk::Buffer,
        format: vk::Format,
    ) -> Option<vk::BufferView> {
        // Create SRV
        let create_info = vk::BufferViewCreateInfo::builder()
            .buffer(buffer)
            .format(format)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let srv = unsafe { self.device.create_buffer_view(&create_info, None) }.ok()?;
        Some(srv)
    }
}
