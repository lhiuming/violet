use glam::u32;

use crate::{
    render_device::{Buffer, RenderDevice},
    render_graph::{PassBuilderTrait, RGHandle, RenderGraphBuilder},
};

use super::div_round_up;

pub fn clear_buffer<'a>(
    rd: &RenderDevice,
    rg: &'a mut RenderGraphBuilder<'_>,
    buffer: RGHandle<Buffer>,
    pass_name: &str,
) {
    let max_group_count_x = rd.physical.properties.limits.max_compute_work_group_count[0] as u64;
    let uint_size = rg.get_buffer_desc(buffer).size / 4;
    if uint_size <= max_group_count_x * 128 {
        rg.new_compute(pass_name)
            .compute_shader("util/clear_buffer.hlsl")
            .rw_buffer("rw_buffer", buffer)
            .push_constant::<u32>(&(uint_size as u32))
            .group_count(div_round_up(uint_size, 128) as u32, 1, 1);
    } else {
        unimplemented!(
            "clear_buffer: buffer size too large ({} > {} * 128)",
            uint_size,
            max_group_count_x
        );
    }
}
