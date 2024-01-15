use ash::vk::{self};
use gpu_allocator::{
    self,
    vulkan::{self, Allocation},
};

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

    // Read/Write in GPU, with fine-tuned usage flags
    pub fn compute_with_usage(size: u64, usage: vk::BufferUsageFlags) -> Self {
        Self {
            size,
            usage,
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
    pub data: *mut u8, // TODO make optional
    pub device_address: Option<vk::DeviceAddress>,

    // Internal
    alloc_index: u16,
}

pub type BufferViewDesc = vk::Format;

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct BufferView {
    pub buffer: Buffer,
    pub desc: BufferViewDesc,
    pub handle: vk::BufferView,
}

impl super::RenderDevice {
    pub fn create_buffer(&mut self, desc: BufferDesc) -> Option<Buffer> {
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

        let mem_req = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        // Allocate memory for ths buffer
        // TODO drop device_memory if later stage failed
        // TODO use a allocator like VMA to do sub allocate
        let memory: vk::DeviceMemory = {
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

        // Map raw memory_property to gpu_allocator::MemoryLocation
        // TODO use better sematics
        let mem_location = if desc.memory_property == vk::MemoryPropertyFlags::DEVICE_LOCAL {
            gpu_allocator::MemoryLocation::GpuOnly
        } else if desc
            .memory_property
            .contains(vk::MemoryPropertyFlags::HOST_CACHED)
        {
            gpu_allocator::MemoryLocation::GpuToCpu
        } else if desc.memory_property.intersects(
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) {
            gpu_allocator::MemoryLocation::CpuToGpu
        } else {
            println!(
                "RenderDevice: unhandled memory property: {:?}. use Unknown",
                desc.memory_property
            );
            gpu_allocator::MemoryLocation::Unknown
        };

        // testing New allocator
        let alloc = self
            .allocator
            .allocate(&vulkan::AllocationCreateDesc {
                name: "unnamed_buffer",
                requirements: mem_req,
                location: mem_location,
                linear: true, // buffer is always linear
                allocation_scheme: vulkan::AllocationScheme::GpuAllocatorManaged, // not dedicated for this object
            })
            .ok()?;

        // Bind
        match unsafe {
            self.device
                .bind_buffer_memory(buffer, alloc.memory(), alloc.offset())
        } {
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
        /*
        let is_mappable = desc
            .memory_property
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE);
        let data = if is_mappable {
            let map_flags = vk::MemoryMapFlags::default(); // dummy parameter
            unsafe {
                self.device
                    .map_memory(alloc.memory(), alloc.offset(), desc.size, map_flags)
            }
            .unwrap() as *mut u8
            alloc.mapped_ptr()
        } else {
            std::ptr::null_mut::<u8>()
        };
        */
        let data = alloc
            .mapped_ptr()
            .map(|p| p.as_ptr() as *mut u8)
            .unwrap_or(std::ptr::null_mut::<u8>());

        let alloc_index = {
            if let Some(index) = self.allocations_free_slot.pop() {
                self.allocations[index as usize] = alloc;
                index
            } else {
                let index = self.allocations.len();
                self.allocations.push(alloc);
                index as u16
            }
        };

        Some(Buffer {
            desc,
            handle: buffer,
            data,
            device_address,
            alloc_index,
        })
    }

    pub fn destroy_buffer(&mut self, buffer: Buffer) {
        let mut alloc = Allocation::default();
        std::mem::swap(
            &mut alloc,
            &mut self.allocations[buffer.alloc_index as usize],
        );

        // Free
        self.allocator
            .free(alloc)
            .expect("failed to free buffer allocation");

        self.allocations_free_slot.push(buffer.alloc_index);

        // Destroy object
        unsafe {
            /*
            if buffer.data != std::ptr::null_mut::<u8>() {
                self.device.unmap_memory(alloc.memory());
            }
            self.device.free_memory(alloc.memory(), None);
             */
            self.device.destroy_buffer(buffer.handle, None);
        }
    }

    pub fn create_buffer_view(&self, buffer: Buffer, desc: BufferViewDesc) -> Option<BufferView> {
        // Create BufferView object
        let create_info = vk::BufferViewCreateInfo::builder()
            .buffer(buffer.handle)
            .format(desc)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let handle = unsafe { self.device.create_buffer_view(&create_info, None) }.ok()?;
        Some(BufferView {
            buffer,
            desc,
            handle,
        })
    }
}
