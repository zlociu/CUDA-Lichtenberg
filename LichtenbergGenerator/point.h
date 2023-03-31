#pragma once
#include "cuda_runtime.h"

struct point_t
{
	int x;
	int y;
	__host__ point_t() {}
	__device__ point_t(int _x, int _y) : x(_x), y(_y) {}

	__host__ __device__ bool isNull() { return (this->x == -1 || this->y == -1); }
	__device__ __inline__ bool isNotNull() { return !this->isNull(); }

	__device__ __inline__ bool operator==(point_t& other) { return this->x == other.x && this->y == other.y; }
	__device__ __inline__ bool operator!=(point_t& other) { return this->x != other.x || this->y != other.y; }
};

//dlugosc krawedzi o konach w punktach p1 i p2
__host__ __device__ float lenght(point_t p1, point_t p2)
{
	return (float)((p2.x - p1.x) * (p2.x - p1.x) * 1.0f + (p2.y - p1.y) * (p2.y - p1.y) * 1.0f);
}