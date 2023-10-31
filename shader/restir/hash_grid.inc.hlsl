#pragma once

#include "../enc.inc.hlsl"
#include "../frame_bindings.hlsl"
#include "../rand.hlsl"

// TODO try prime bucket size?
#if 1
#define HASH_GRID_NUM_BUCKETS 65536 // 64 * 1024
#define HASH_GRID_NUM_BUCKETS_BITMASK 0xFFFF
#else
#define HASH_GRID_NUM_BUCKETS (65536 / 2)
#define HASH_GRID_NUM_BUCKETS_BITMASK 0x7FFF
#endif

// TODO experiment with different capacity, or open addressing (with lazy deletion?) instead of chaining? (lazy deletion seems hard to run parallel)
#define HASH_GRID_BUCKET_SIZE 16
#define HASH_GRID_BUCKET_SIZE_BITSHIFT 4

#define HASH_GRID_NUM_CELLS (HASH_GRID_NUM_BUCKETS * HASH_GRID_BUCKET_SIZE)

#define HASH_GRID_BASE_CELL_SIZE (1.0 / 8.0)
#define HASH_GRID_BASE_CELL_SIZE_INV (1.0 / HASH_GRID_BASE_CELL_SIZE)

#define HASH_GRID_MAX_NUM_QUERIES (128 * 1024)
#define HASH_GRID_MAX_NUM_QUERIES_BITSHIFT 17

//
// Vertex Descriptor and Quantization
//

uint hash_grid_cell_lod(float3 pos, float3 camera_pos)
{
    float3 delta = pos - camera_pos;
    float dist_sq = dot(delta, delta);

    #if 1
    // clip-map style log2(distance) LOD
    float lod = max(0.5 * log2(dist_sq), 0.0); 
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
        // quanize pos before calculating distance-based lod, to reduce overlapping hash cells at LOD boundary
        // TODO check load factor
        float3 lod_pos = pos;
        #if 0
        lod_pos = round(pos * 4.0) * 0.25;
        #endif

        // TODO adaptive cell size (LOD) by ray length
        uint lod = hash_grid_cell_lod(lod_pos, frame_params.view_pos.xyz);

        // quantize
        float cell_size = HASH_GRID_BASE_CELL_SIZE * float(1u << lod);
        #if 1
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
        #if 0
        uint hash = pcg_hash(pos_u32.x + pcg_hash(pos_u32.y + pcg_hash(pos_u32.z)));
        hash = pcg_hash(hash + lod);
        #else
        uint hash = XXHash32::init(pos_u32.y).add(lod).add(pos_u32.z).add(pos_u32.x).eval();
        #endif
        // NOTE: zero is reserved to indicate empty cell
        return hash | 0x1;
    }
};

//
// Hash Grid Storage
//

// TODO break checksum into separate buffer, to reduce cache trashing?
struct HashGridCell
{
    uint checksum; // "hash2" / "verification hash"
    float3 radiance;

    static uint3 radiance_scale(float3 radiance, float weight)
    {
        return uint3(round(radiance * (4096.0 * weight)));
    }

    static float3 radiance_unscale(uint3 radiance, float weight)
    {
        return float3(radiance) / (4096.0 * weight);
    }

    static HashGridCell empty() 
    {
        HashGridCell ret = { 0, float3(0.0, 0.0, 0.0) };
        return ret;
    }
};

//bool hash_grid_find(StructuredBuffer<HashGridCell> buffer, uint hash, uint checksum, out HashGridCell cell, out uint addr)
template<typename CellBuffer>
bool hash_grid_find(CellBuffer buffer, uint hash, uint checksum, out HashGridCell cell, out uint addr)
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
            else
            {
                // a different cell is inserted; continue probing.
            }
        }
    }

    return false;
}

struct HashGridQuery {
    float3 hit_position;
    uint hit_normal_encoded;
    uint cell_hash;
    uint cell_checksum;

    static uint encode_normal(float3 n)
    {
        return normal_encode_oct_u32(n);
    }

    static HashGridQuery create(uint hash, uint checksum, float3 pos, float3 normal)
    {
        HashGridQuery ret = { pos, encode_normal(normal), hash, checksum };
        return ret;
    }

    float3 hit_normal() 
    {
        return normal_decode_oct_u32(hit_normal_encoded);
    }
};