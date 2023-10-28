#pragma once

// Cemera position
#include "../frame_bindings.hlsl"
// Hash functions
#include "../rand.hlsl"

// TODO try prime bucket size?
#define HASH_GRID_NUM_BUCKETS 65536
#define HASH_GRID_NUM_BUCKETS_BITMASK 0xFFFF

// TODO experiment with different capacity, or open addressing (with lazy deletion?) instead of chaining? (lazy deletion seems hard to run parallel)
#define HASH_GRID_BUCKET_SIZE 16
#define HASH_GRID_BUCKET_SIZE_BITSHIFT 4

#define HASH_GRID_NUM_CELLS (HASH_GRID_NUM_BUCKETS * HASH_GRID_BUCKET_SIZE)

#define HASH_GRID_BASE_CELL_SIZE (1.0 / 16.0)
#define HASH_GRID_BASE_CELL_SIZE_INV (1.0 / HASH_GRID_BASE_CELL_SIZE)

struct QueryRecord
{
    float3 pos;
    float3 dir_in;

    uint4 encode()
    {
        uint3 pos_u32 = asuint(pos);
        return uint4(pos_u32, 0);
    }

    static QueryRecord decode(uint4 data)
    {
        float3 pos = asfloat(data.xyz);
        QueryRecord ret = { pos, float3(0, 0, 0) };
        return ret;
    }
};

uint hash_grid_cell_lod(float3 pos, float3 camera_pos)
{
    float3 delta = pos - camera_pos;
    float dist_sq = dot(delta, delta);

    #if 0
    // clip-map style log2(distance) LOD
    //float lod = max(0.5 * log2(dist_sq), 0.0); 
    #else
    // somewhat evenly distributed on screen (tuned on current FOV (90 deg))
    float lod = max(0.38 * log2(dist_sq), 0.0); 
    #endif

    return uint(lod);
}

// Key to access the cache cell.
// Spatial and directional filtering is achieved by quantization.
struct VertexDescriptor
{
    int3 pos; // quantized hit point
    float3 dir_in; // incoming directin / "view" direction for hit point
    uint lod;

    static VertexDescriptor create(float3 pos, float3 dir_in)
    {
        // quantize
        // TODO adaptive cell size (LOD) by ray length
        uint lod = hash_grid_cell_lod(pos, frame_params.view_pos.xyz);
        #if 1
        float cell_size = HASH_GRID_BASE_CELL_SIZE * float(1u << lod);
        int3 pos_quant = int3(floor(pos / cell_size));
        #else
        int3 pos_quant = int3(floor(pos * HASH_GRID_BASE_CELL_SIZE_INV)) >> lod;
        #endif

        VertexDescriptor vert = { pos_quant, dir_in, lod };
        return vert;
    }

    // Hash used to access the bucket
    uint hash()
    {
        // TODO try pcg3d16?
        uint3 pos_u32 = asuint(pos);
        uint hash = XXHash32::init(pos_u32.x).add(pos_u32.y).add(pos_u32.z).add(lod).eval();
        return hash & HASH_GRID_NUM_BUCKETS_BITMASK;
    }

    // Hash used for linear probing
    uint checksum()
    {
        uint3 pos_u32 = asuint(pos);
        uint hash = pcg_hash(pos_u32.x + pcg_hash(pos_u32.y + pcg_hash(pos_u32.z)));
        // NOTE: zero is reserved to indicate empty cell
        return hash | 0x1;
    }
};

// TODO break checksum into separate buffer, to reduce cache trashing?
struct HashGridCell
{
    uint checksum; // "hash2" / "verification hash"
    float3 radiance; 

    static HashGridCell empty() 
    {
        HashGridCell ret = { 0, float3(0, 0, 0) };
        return ret;
    }
};

bool hash_grid_find(StructuredBuffer<HashGridCell> buffer, uint hash, uint checksum, out HashGridCell cell, out uint addr)
{
    // linear probing in the bucket
    uint addr_beg = hash << HASH_GRID_BUCKET_SIZE_BITSHIFT;
    uint addr_end = addr_beg + HASH_GRID_BUCKET_SIZE;
    addr = addr_beg;
    for ( ; addr < addr_end; ++addr)
    {
        cell = buffer[addr];
        if (cell.checksum == checksum)
        {
            return true;
        }
        if (cell.checksum == 0)
        {
            // early out; the bucket is compact
            break;
        }
    }

    return false;
}

bool hash_grid_find_or_insert(RWStructuredBuffer<HashGridCell> buffer, uint hash, uint checksum, out uint addr)
{
    // linear probing in the bucket
    uint addr_beg = hash << HASH_GRID_BUCKET_SIZE_BITSHIFT;
    uint addr_end = addr_beg + HASH_GRID_BUCKET_SIZE;
    addr = addr_beg;
    for ( ; addr < addr_end; ++addr)
    {
        uint slot_checksum = buffer[addr].checksum;
        if (slot_checksum == checksum)
        {
            // found existing cell
            return true;
        }
        if (slot_checksum == 0)
        {
            // found an empty slot in the bucket => rest of the bucket is empty
            // try to insert the cell
            InterlockedCompareExchange(buffer[addr].checksum, 0, checksum, slot_checksum);
            // NOTE: another thread may have inserted the cell with a same checksum
            bool insert_by_this = (slot_checksum == 0);
            bool insert_by_other = (slot_checksum == checksum);
            if ( insert_by_this || insert_by_other )
            {
                // insert successfully (maybe by another thread)
                return true;
            }
        }
    }

    return false;
}