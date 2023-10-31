/*
 * For each quried cell, randomly select one query from (possibly) multiple queries, which are scatterred into the query buffer during ray-tracing passes.
 */

#include "../rand.hlsl"

#include "hash_grid.inc.hlsl"

StructuredBuffer<HashGridQuery> hash_grid_query_buffer;
StructuredBuffer<uint> hash_grid_query_counter_buffer;

// TODO store checksum in seperate buffer
RWStructuredBuffer<HashGridCell> rw_hash_grid_storage_buffer;
RWStructuredBuffer<uint> rw_hash_grid_query_selection_buffer;

[[vk::push_constant]]
struct PC
{
    uint frame_hash;
} pc;

[numthreads(32, 1, 1)]
void main(uint dispatch_id : SV_DispatchThreadID)
{
    if (dispatch_id >= hash_grid_query_counter_buffer[0])
    {
        return;
    }

    uint query_index = dispatch_id;
    HashGridQuery query = hash_grid_query_buffer[query_index];

    // Ensure a hash grid cell for this query
    uint cell_addr;
    if ( !hash_grid_find_or_insert(rw_hash_grid_storage_buffer, query.cell_hash, query.cell_checksum, cell_addr) )
    {
        return;
    }

    // Randomly select one query (from possible multiple queries) for the cell
    uint lcg_state = jenkins_hash( dispatch_id ^ pc.frame_hash );
    uint rand_u32 = lcg_xorshift(lcg_state);

    // the uint32 in selection buffer is devided to three fields:
    //   uint[ 0:17): query_index, pointer to the allocated query
    //   uint[17:18): reserved to indicate non-null slot.
    //   uint[18:32): random score, put in MSBs, to do randomized selection
    // NOTE: HASH_GRID_MAX_NUM_QUERIES_BITSHIFT == 17 
    uint reserved_bit = 0x1 << 31;
    uint score_quantized = rand_u32 & (0xFFFFFFFFu << (HASH_GRID_MAX_NUM_QUERIES_BITSHIFT + 1));
    uint compare_word = score_quantized | reserved_bit | query_index;

    // Compare and swap
    uint original_word;
    InterlockedMax(rw_hash_grid_query_selection_buffer[cell_addr], compare_word, original_word);
}