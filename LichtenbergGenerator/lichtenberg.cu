#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "./helpers.h"
#include "./point.h"

#include <stdio.h>
#include <fstream>
#include <nvgraph.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <time.h>


typedef enum 
{
	LICHTENBERG_STATUS_SUCCESS = 0,
	LICHTENBERG_STATUS_DIM_SQUARE_SIZE = 1, // picture size is not multiple of square size
	LICHTENBERG_STATUS_CUDA_ERROR = 2,
	LICHTENBERG_STATUS_NV_GRAPH_ERROR = 3,
	LICHTENBERG_STATUS_CURAND_ERROR = 4,
	LICHTENBERG_STATUS_OTHER_ERROR = 99
} lichtenbergStatus_t;


struct inputArgs_t
{
	int startPos;
	int verticesCount;
	char* lightningColor;
	char* backgroundColor;
};

__global__ void setup_kernel(curandState* state)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	curand_init(clock() ^ offset, 1, 0, &state[offset]);
}

//tab - pixels table, p - propability
__global__ void createVertices(point_t* crate_tab, const float p, curandState* cu, const int squareSize)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	curandState rnd = cu[offset];
	unsigned int* lista = new unsigned int[3];
	lista[0] = curand(&rnd); //x
	lista[1] = curand(&rnd); //y
	lista[2] = curand(&rnd) & 0x03FF; //p = rng & 1024
	cu[offset] = rnd;

	lista[0] = lista[0] % (squareSize - 2) + 1;
	lista[1] = lista[1] % (squareSize - 2) + 1;
	int x2 = x * squareSize;
	int y2 = y * squareSize;

	if ((lista[2]) < (int)(p * 1024.f))
	{
		crate_tab[offset] = point_t{ lista[0] + x2, lista[1] + y2 };
	}
	else
	{
		crate_tab[offset] = { -1, -1 };
	}
}

/// <summary>
/// 
/// </summary>
/// <param name="vertices"></param>
/// <param name="edges"></param>
/// <param name="offsets">- how many edges for each vertex</param>
/// <param name="weights">- edge weight</param>
/// <param name="seed">- seed to determine if '\' edge or '/'</param>
/// <returns></returns>
__global__ void createEdges(point_t* vertices, int* edges, int* offsets, float* weights, int seed)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x; //column
	int y = threadIdx.y + blockIdx.y * blockDim.y; //row
	int offset = x + y * blockDim.x * gridDim.x;

	int edges_cnt = 0;

	bool top, bottom, left, right;
	top = bottom = left = right = false;

	if (vertices[offset].isNull())
	{
		offsets[offset] = edges_cnt;
		return;
	}

	// index of second vertex, with which we create edge
	int second_index = offset - blockDim.x * gridDim.x;

	// top
	if (y > 0 && vertices[second_index].isNotNull())
	{
		edges[offset * 8 + edges_cnt] = second_index;
		weights[offset * 8 + edges_cnt] = lenght(vertices[offset], vertices[second_index]);
		edges_cnt++;
		top = true;
	}

	second_index = offset - 1;

	// left
	if (x > 0 && vertices[second_index].isNotNull())
	{
		edges[offset * 8 + edges_cnt] = second_index;
		weights[offset * 8 + edges_cnt] = lenght(vertices[offset], vertices[second_index]);
		edges_cnt++;
		left = true;
	}

	second_index = offset + 1;

	// right
	if (x < blockDim.x * gridDim.x - 1 && vertices[second_index].isNotNull())
	{
		edges[offset * 8 + edges_cnt] = second_index;
		weights[offset * 8 + edges_cnt] = lenght(vertices[offset], vertices[second_index]);
		edges_cnt++;
		right = true;
	}

	second_index = offset + blockDim.x * gridDim.x;

	// bottom
	if (y < blockDim.y * gridDim.y - 1 && vertices[second_index].isNotNull())
	{
		edges[offset * 8 + edges_cnt] = second_index;
		weights[offset * 8 + edges_cnt] = lenght(vertices[offset], vertices[second_index]);
		edges_cnt++;
		bottom = true;
	}

	second_index = offset - blockDim.x * gridDim.x - 1;

	// top left
	if (x > 0 && y > 0 && vertices[second_index].isNotNull())
	{
		if (top && left)
		{
			if ((seed & (x - 1) & 1) == 1)
			{
				edges[offset * 8 + edges_cnt] = second_index;
				weights[offset * 8 + edges_cnt] = lenght(vertices[offset], vertices[second_index]);
				edges_cnt++;
			}
		}
		else
		{
			edges[offset * 8 + edges_cnt] = second_index;
			weights[offset * 8 + edges_cnt] = lenght(vertices[offset], vertices[second_index]);
			edges_cnt++;
		}
	}

	second_index = offset - blockDim.x * gridDim.x + 1;

	// top right
	if (x < blockDim.x * gridDim.x - 1 && y > 0 && vertices[second_index].isNotNull())
	{
		if (top && right)
		{
			if ((seed & (x) & 1) == 1)
			{
				edges[offset * 8 + edges_cnt] = second_index;
				weights[offset * 8 + edges_cnt] = lenght(vertices[offset], vertices[second_index]);
				edges_cnt++;
			}
		}
		else
		{
			edges[offset * 8 + edges_cnt] = second_index;
			weights[offset * 8 + edges_cnt] = lenght(vertices[offset], vertices[second_index]);
			edges_cnt++;
		}
	}

	second_index = offset + blockDim.x * gridDim.x - 1;

	// bottom left
	if (x > 0 && y < blockDim.y * gridDim.y - 1 && vertices[second_index].isNotNull())
	{
		if (bottom && left)
		{
			if ((seed & (x - 1) & 1) == 1)
			{
				edges[offset * 8 + edges_cnt] = second_index;
				weights[offset * 8 + edges_cnt] = lenght(vertices[offset], vertices[second_index]);
				edges_cnt++;
			}
		}
		else
		{
			edges[offset * 8 + edges_cnt] = second_index;
			weights[offset * 8 + edges_cnt] = lenght(vertices[offset], vertices[second_index]);
			edges_cnt++;
		}
	}

	second_index = offset + blockDim.x * gridDim.x + 1;

	// bottom right
	if (x < blockDim.x * gridDim.x - 1 && y < blockDim.y * gridDim.y - 1 && vertices[second_index].isNotNull())
	{
		if (bottom && right)
		{
			if ((seed & (x) & 1) == 1)
			{
				edges[offset * 8 + edges_cnt] = second_index;
				weights[offset * 8 + edges_cnt] = lenght(vertices[offset], vertices[second_index]);
				edges_cnt++;
			}
		}
		else
		{
			edges[offset * 8 + edges_cnt] = second_index;
			weights[offset * 8 + edges_cnt] = lenght(vertices[offset], vertices[second_index]);
			edges_cnt++;
		}
	}

	offsets[offset] = edges_cnt;
}

int adjustVerticesEdges(
	point_t* vertices_src, point_t* vertices_dest,
	const int* edges_src, int* edges_dest,
	const int* offsets_src, int* offsets_dest,
	const float* weights_src, float* weights_dest,
	const int crateSizeX, const int crateSizeY)
{
	int vert_offset = 0;
	int edge_offset = 0;
	int null_idx = 0;

	int* shift_idx = (int*)malloc(crateSizeX * crateSizeY * sizeof(int)); //if point is before or after how many null points
	for (int i = 0; i < crateSizeX * crateSizeY; i++)
	{
		shift_idx[i] = null_idx;
		null_idx += vertices_src[i].isNull();
	}

	for (int i = 0; i < crateSizeX * crateSizeY; i++)
	{
		if (vertices_src[i].isNull()) continue;

		int edge_cnt = offsets_src[i];

		offsets_dest[vert_offset] = edge_offset;
		vertices_dest[vert_offset] = vertices_src[i];
		vert_offset++;

		for (int e = 0; e < edge_cnt; e++)
		{
			edges_dest[edge_offset + e] = edges_src[8 * i + e] - shift_idx[edges_src[8 * i + e]];
			weights_dest[edge_offset + e] = weights_src[8 * i + e];
		}

		edge_offset += edge_cnt;
	}

	offsets_dest[vert_offset] = edge_offset;
	return vert_offset;
}

void setWidth(int* offsets, int* edges, float* weights, int* width, float* sssh_1, int startPoint, int numVertices, int cntNum)
{
	int num = (rand() % cntNum / 10) + cntNum; //number of ending vertices
	int idx;
	int dlugosc;
	bool* zajete = (bool*)calloc(numVertices, sizeof(bool));
	for (int i = 0; i < num; i++)
	{
		idx = rand() % numVertices;
		if (zajete[idx] == false)
		{
			zajete[idx] = true;
			dlugosc = (int)sssh_1[idx];

			if (dlugosc > 10'000'000) continue;
			
			while (dlugosc > 0)
			{
				for (int k = offsets[idx]; k < offsets[idx + 1]; k++)
				{
					if (sssh_1[edges[k]] + (int)weights[k] == dlugosc)
					{
						width[k]++;
						dlugosc = (int)sssh_1[edges[k]];
						idx = edges[k];
						break;
					}
				}
			}
		}
	}
}

const int saveXML(
	unsigned char* result,
	point_t* points,
	int* offsets,
	int* edges,
	int* width,
	int cntVertices,
	int dimX,
	int dimY,
	char* foreground,
	char* background = "White")
{
	int offset = 0;
	char* buffer = (char*)malloc(150 * sizeof(char));
	int n = sprintf(buffer, "<svg height=\"100%%\" width=\"100%%\" viewBox=\"-16 -16 %d %d \" xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\">\n", dimX + 32, dimY + 32);
	memcpy(result, buffer, n);
	offset += n;

	n = sprintf(buffer, "<style>line{ stroke:%s; stroke-linecap:round; }</style>\n", foreground);
	memcpy(result + offset, buffer, n);
	offset += n;

	n = sprintf(buffer, "<rect x=\"-16\" y=\"-16\" width=\"100%%\" height=\"100%%\" style=\"fill:%s; stroke-width:0\"/>\n", background);
	memcpy(result + offset, buffer, n);
	offset += n;

	for (int i = 0; i < cntVertices; i++)
	{
		for (int k = offsets[i]; k < offsets[i + 1]; k++)
		{
			if (width[k] > 0)
			{
				n = sprintf(buffer, "<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" style=\"stroke-width:%.1f\"/>\n", points[i].x, points[i].y, points[edges[k]].x, points[edges[k]].y, (sqrtf(width[k])));
				memcpy(result + offset, buffer, n);
				offset += n;
			}
		}
	}

	memcpy(result + offset, "</svg>\n", 8);
	return offset + 7;
}

struct DataBlock {
	point_t* vertices; // points in table 
	curandState* random; //rng
	int* edges; 
	int* offsets;  // how many edges has each vertex
	float* weights; //weights (lenghts) between vertices
};

/// <summary>
/// 
/// </summary>
/// <param name="dimX"></param>
/// <param name="dimY"></param>
/// <param name="squareSize"></param>
/// <returns>0 - good</returns>
extern "C" __declspec (dllexport) int createLichtenberg(
	int dimX,
	int dimY,
	int squareSize,
	inputArgs_t inputArgs,
	unsigned char* result,
	int* len)
{
	if (dimX % 32 != 0 || dimY % 32 != 0)
	{
		return LICHTENBERG_STATUS_DIM_SQUARE_SIZE;
	}

	int crateSizeX = dimX / squareSize;
	int crateSizeY = dimY / squareSize;

	srand(time(0));
	// -------------------< bitmap >---------------------- 
	DataBlock   data;

	// -------------------< graph >---------------------- 
	point_t* vertices; 
	vertices = (point_t*)malloc(crateSizeX * crateSizeY * sizeof(point_t));
	point_t* vertices2; 
	vertices2 = (point_t*)malloc(crateSizeX * crateSizeY * sizeof(point_t));

	int* offsets;
	offsets = (int*)malloc(crateSizeX * crateSizeY * sizeof(int));
	int* offsets2;
	offsets2 = (int*)malloc(crateSizeX * crateSizeY * sizeof(int));

	int* edges; 
	edges = (int*)malloc(crateSizeX * crateSizeY * 8 * sizeof(int));
	int* edges2; 
	edges2 = (int*)malloc(crateSizeX * crateSizeY * 8 * sizeof(int));

	float* weights; 
	weights = (float*)malloc(crateSizeX * crateSizeY * 8 * sizeof(float));
	float* weights2; 
	weights2 = (float*)malloc(crateSizeX * crateSizeY * 8 * sizeof(float));

	int* width;
	width = (int*)calloc(crateSizeX * crateSizeY * 8, sizeof(int)); //grubosc krawedzi

	int number_vertices = 0;
	int number_edges = 0;

	// -------------------< malloc >---------------------- 
	HANDLE_ERROR(cudaMalloc((void**)&data.random, crateSizeX * crateSizeY * sizeof(curandState)));  
	HANDLE_ERROR(cudaMalloc((void**)&data.vertices, crateSizeX * crateSizeY * sizeof(point_t))); 
	HANDLE_ERROR(cudaMalloc((void**)&data.edges, crateSizeX * crateSizeY * 8 * sizeof(int))); 
	HANDLE_ERROR(cudaMalloc((void**)&data.weights, crateSizeX * crateSizeY * 8 * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&data.offsets, crateSizeX * crateSizeY * sizeof(int)));

	dim3 grid(crateSizeX >> 3, crateSizeY >> 3); //size/8
	dim3 threads(8, 8);

	// -------------------< CUDA >---------------------- 
	setup_kernel << <grid, threads >> > (data.random);
	createVertices << <grid, threads >> > (data.vertices, 0.93f, data.random, squareSize);
	createEdges << <grid, threads >> > (data.vertices, data.edges, data.offsets, data.weights, rand());

	HANDLE_ERROR(cudaMemcpy(vertices, data.vertices, crateSizeX * crateSizeY * sizeof(point_t), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(edges, data.edges, crateSizeX * crateSizeY * 8 * sizeof(int), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(weights, data.weights, crateSizeX * crateSizeY * 8 * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(offsets, data.offsets, crateSizeX * crateSizeY * sizeof(int), cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaFree(data.vertices));
	HANDLE_ERROR(cudaFree(data.edges));
	HANDLE_ERROR(cudaFree(data.weights));
	HANDLE_ERROR(cudaFree(data.offsets));

	// --------------------< CPU >----------------------
	number_vertices = adjustVerticesEdges(
		vertices, vertices2,
		edges, edges2,
		offsets, offsets2,
		weights, weights2,
		crateSizeX, crateSizeY);

	number_edges = offsets2[number_vertices];

	free(weights); free(vertices);
	free(edges); free(offsets);

	// -------------------< nvGraph >---------------------- 
	const size_t  n_vertex = number_vertices, n_edge = number_edges, vertex_numsets = 1, edge_numsets = 1;
	float* sssp_1_h; //sssp results
	void** vertex_dim; //1
	// nvgraph variables
	nvgraphHandle_t handle;
	nvgraphGraphDescr_t graph;
	nvgraphCSCTopology32I_t CSC_input;
	cudaDataType_t edge_dimT = CUDA_R_32F;
	cudaDataType_t* vertex_dimT;
	// Init host data
	sssp_1_h = (float*)malloc(n_vertex * sizeof(float));
	vertex_dim = (void**)malloc(vertex_numsets * sizeof(void*));
	vertex_dimT = (cudaDataType_t*)malloc(vertex_numsets * sizeof(cudaDataType_t));
	CSC_input = (nvgraphCSCTopology32I_t)malloc(sizeof(struct nvgraphCSCTopology32I_st));
	vertex_dim[0] = (void*)sssp_1_h; vertex_dimT[0] = CUDA_R_32F;
	float* weights_h; //weights
	int* destination_offsets_h; //offset 
	int* source_indices_h; //(edges are directed, need to do: 2->1 i 1->2)

	weights_h = (float*)malloc(n_edge * sizeof(float));
	destination_offsets_h = (int*)malloc((n_vertex + 1) * sizeof(int));
	source_indices_h = (int*)malloc((n_edge + 1) * sizeof(int));

	memcpy(weights_h, weights2, n_edge * sizeof(float));
	memcpy(destination_offsets_h, offsets2, n_vertex * sizeof(int));
	destination_offsets_h[n_vertex] = n_edge;
	memcpy(source_indices_h, edges2, n_edge * sizeof(int));

	CHECK(nvgraphCreate(&handle));
	CHECK(nvgraphCreateGraphDescr(handle, &graph));
	CSC_input->nvertices = n_vertex; CSC_input->nedges = n_edge;
	CSC_input->destination_offsets = destination_offsets_h;
	CSC_input->source_indices = source_indices_h;
	// Set graph connectivity and properties (tranfers)
	CHECK(nvgraphSetGraphStructure(handle, graph, (void*)CSC_input, NVGRAPH_CSC_32));
	CHECK(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
	CHECK(nvgraphAllocateEdgeData(handle, graph, edge_numsets, &edge_dimT));
	CHECK(nvgraphSetEdgeData(handle, graph, (void*)weights_h, 0));
	// Solve
	int source_vert = inputArgs.startPos;
	if (source_vert > n_vertex) source_vert %= n_vertex;
	CHECK(nvgraphSssp(handle, graph, 0, &source_vert, 0));
	// Get and print result
	CHECK(nvgraphGetVertexData(handle, graph, (void*)sssp_1_h, 0));

	// -------------------< CPU >---------------------- 
	setWidth(offsets2, edges2, weights2, width, sssp_1_h, source_vert, n_vertex, inputArgs.verticesCount);
	*len = saveXML(result, vertices2, offsets2, edges2, width, n_vertex, dimX, dimY, inputArgs.lightningColor, inputArgs.backgroundColor);

	// -------------------< clean >----------------------  
	free(sssp_1_h); free(vertex_dim);
	free(vertex_dimT); free(CSC_input);
	CHECK(nvgraphDestroyGraphDescr(handle, graph));
	CHECK(nvgraphDestroy(handle));

	free(offsets2); free(edges2);
	free(weights2); free(width);

	return LICHTENBERG_STATUS_SUCCESS;
}