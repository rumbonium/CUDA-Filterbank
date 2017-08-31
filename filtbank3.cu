/*
 ============================================================================
 Name        : filtbank3.cu
 Author      : Hyrum
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cufft.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>
#include <complex>
#include "filtb.h"
#include "filtd.h"
#include "filt7.h"

const int NUMCHANNELS = 100;
const int FILTLENGTH = COL + NUMCHANNELS - (COL % NUMCHANNELS);
const int SLICESIZE = 4000000;
const int BLOCKX = 2;
const int BLOCKY = NUMCHANNELS;
const int THREADX = (FILTLENGTH / NUMCHANNELS) * 2;
const int CS = SLICESIZE * BLOCKX;

const int LD = CDL;
const int BD = 400;
const int SD = 200;
const int LLD = 140;
const int DEL = (LD-1)/2;
const int TD = (LLD*2);
const int CSD = CS/NUMCHANNELS;

const int D7 = 5;
const int L7 = CL7;
const int B7 = 200;
const int S7 = 200;
const int LL7 = L7 + D7 - (L7 % D7);
const int T7 = LL7;
const int CS7 = CSD/2;

const int OS = CS7/D7;

union semun {
    int              val;    /* Value for SETVAL */
    struct semid_ds *buf;    /* Buffer for IPC_STAT, IPC_SET */
    unsigned short  *array;  /* Array for GETALL, SETALL */
    struct seminfo  *__buf;  /* Buffer for IPC_INFO (Linux-specific) */
};

class meminfo {
 public:
  int siz; // number of bytes per element of shared memory
  int chk; // number of elements in each chunk of shared memory
  int csz; // number of bytes in each chunk of shared memory
  int num; // number of chunks of shared memory
  int ele; // total number of elements in shraed memory
  int tot; // total number of bytes in shared memory
  meminfo(int sz, int ck, int nm) : siz(sz), chk(ck), num(nm) {
    csz = siz * chk;
    ele = chk * num;
    tot = ele * siz;
  }
  ~meminfo() {}
};

#define DIE(msg) { perror(msg); return 1; }

__global__ void tempBufSet(float* temp, int LL, int size)
/* prepares a temporary buffer for the next filter.*/
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < 2*LL) { temp[tid] = temp[tid + size]; }
}

__global__ void filtbank(float* COEF, float* in, float* out, int SS, int FL, int NC)
{
	extern __shared__ float H[];
	int biy = blockIdx.y; // 0 - 99
	int tid = threadIdx.x; // 0 - 101
	if (tid < FL) {H[tid] = COEF[(tid*NC) + biy];} // NC = 100, FL = 51
	__syncthreads();

	int bix = blockIdx.x; // 0 - 1
	int z = tid % 2;//real or imaginary data
	int dt = tid / 2;//which accumulator you are working with
	int slice_start = (bix * SS) + z + ((NC - 1 - biy)*2);
	int slice_stop = ((bix + 1) * SS) + z + (2 * FL * NC);
	float x;
	int k;
	float acc = 0;
	int time = 0;
	for(int ind = slice_start; ind < slice_stop; ind += 2*NC)
	{
		x = in[ind];
		k = (dt - time + FL) % FL;
		acc += H[k] * x;
		if (time == dt)
		{
			if (ind >= slice_start + (2*FL*NC)) { out[ind + 2*(2*biy - (NC-1)) - (2*FL*NC)] = acc;}
			acc = 0;
		}
		time = (time + 1) % FL;
	}
}

__global__ void ddfilt(float* COEF, float* in, float* diffout, float* delout, int LL, int slice_size, int delay, int offset)
//remember to add offset
{
	extern __shared__ float H[];
	int cid = threadIdx.x;
	if (cid < LL) { H[cid] = COEF[cid]; }

	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int slice_start = slice_size * bid;
	int slice_stop = (slice_size * (bid + 1)) + (2 * LL);
	float x;
	int m;
	bool ready;
	int z = tid % 2;
	int dt = tid / 2;
	int time = 0;
	float y = 0;
	float yy = 0;
	for(int ix = (slice_start + z); ix < slice_stop; ix+=2)
	{
		ready = (ix >= ((2 * LL) + slice_start));
		x = in[ix];
		m = (dt - time + LL) % LL;
		y += H[m] * x;
		if(m == delay) yy = x;
		if(time == dt)
		{
			if(ready)
			{
				diffout[(((ix/2)-LL)*2)+z + (2*offset)] = y;
				delout[(((ix/2)-LL)*2)+z + (2*offset)] = yy;
			}
			y = 0;
		}
		time = (time + 1) % LL;
	}
}

__global__ void crossmult(float* diffin, float* delin, float* out, int size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int idx = tid * 2;
	int num = blockDim.x * gridDim.x;
	int inc = num * 2;
	while(idx < size)
	{
		out[idx/2] = (delin[idx] * diffin[idx + 1]) - (delin[idx + 1] * diffin[idx]);
		//out[idx/2] = (delin[idx+1] * diffin[idx]) - (delin[idx] * diffin[idx+1]);
		idx += inc;
	}
}

__global__ void oldTempBufSet(float* temp, int LL, int size)
/* prepares a temporary buffer for the next filter.*/
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < LL) { temp[tid] = temp[tid + size]; }
}

__global__ void deciconvo(float* COEF, float* inbuf, float* dbuf, int D, int LL, int numThreads, int sliceSize, int nextOff)
{
	extern __shared__ float H[];
	int cid = threadIdx.x;
	while (cid < LL) { H[cid] = COEF[cid]; cid += numThreads;}

	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int slice_start = sliceSize * bid;
	int slice_stop  = (sliceSize * (bid + 1)) + LL;
	float x;
	int m;
	int time = 0;
	float y = 0;
	for(int ix = slice_start; ix < slice_stop; ix++) {
		x = inbuf[ix];
		m = (tid * D) - time + LL;
		y += H[m % LL] * x;
		if(time == tid*D)
		{
			if(ix >= (LL+slice_start)) { dbuf[((ix - LL) / D) + nextOff] = y; }
			y = 0;
		}
		time = (time + 1) % LL;
	}
}

__global__ void betterMemcpy(float* in, float* out, int num_elements)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	while(tid < num_elements)
	{
		out[tid] = in[tid];
		tid += (blockDim.x * gridDim.x);
	}
}

void setup_filter(float *A, const float *G, int L, int LL)
// A: host coefficients, G: pointer from filt.h file
{
	memset(A,0,sizeof(float)*LL);
	for(int i=0; i< L; i++) { A[i] = 20*G[i]; }
}

int main(int argc, char **argv)
{
	int tick = 0;

	cudaError_t err = cudaSuccess;
	cufftResult_t err2;
	float *h_out, *h_COB, *h_COD, *h_CO7;
	float *d_CHUNK, *d_tempbfft, *d_tempfftd, *d_diff, *d_del, *d_temp7, *d_out, *d_COB, *d_COD, *d_CO7, *d_test;
	cufftHandle plan;
	if(cufftCreate(&plan) != CUFFT_SUCCESS) {fprintf(stderr, "CUFFT error: Plan creation failed\n"); exit(EXIT_FAILURE);}
	dim3 numBlocks(BLOCKX, BLOCKY);

	int semid = 0;
	int old_semid = semid;

	meminfo smi(sizeof(float), CS, 50);
	int shmid;
	key_t shm_key = 48879;
	float *shm, *s;
	if ((shmid = shmget(shm_key, smi.tot, IPC_CREAT | 0666)) == -1) DIE("shmget");
	if ((shm = (float*)shmat(shmid, NULL, 0)) == (void*) -1) DIE("shmat");
	s = shm;

	int marker;
	key_t marker_key = 15;
	union semun markerset;
	markerset.val = 1;
	if((marker = semget(marker_key, 1, IPC_CREAT | 0666)) == -1) DIE("semget");
	if(semctl(marker,0,SETVAL,markerset) == -1) DIE("semctl");

	int sempty;
	key_t sempty_key = 64206;
	union semun semptyset;
	int sfull;
	key_t sfull_key = 57005;
	union semun sfullset;
	if ((sempty = semget(sempty_key, smi.num, IPC_CREAT | 0666)) == -1) DIE("semget");
	if ((sfull = semget(sfull_key, smi.num, IPC_CREAT | 0666)) == -1) DIE("semget");

	float station = 98.1;
	if(argc > 1) station = atof(argv[1]);
	int station_num;
	if(station >= 98.1) {station_num = round((station - 98.1)/0.2);}
	else{station_num = round((station - 78.1)/0.2);}
	fprintf(stderr, "station: %d\n", station_num);
	int numchannels = NUMCHANNELS;
	int rank = 1;
	int inembed = CS/2;
	int istride = 1;
	int idist = NUMCHANNELS;
	int onembed = CS/2;
	int ostride = CS/(2*NUMCHANNELS);
	int odist = 1;
	int batch = CS/(2*NUMCHANNELS);
	if((err2 = cufftPlanMany(&plan, rank, &numchannels, &inembed,
					istride, idist, &onembed, ostride,
					odist, CUFFT_C2C, batch)) != CUFFT_SUCCESS)
	{fprintf(stderr, "CUFFT error: Plan design failed: %d\n", err2); exit(EXIT_FAILURE);}

	h_out = (float*)malloc(OS*sizeof(float));
	h_COB = (float*)malloc(FILTLENGTH*sizeof(float));
	h_COD = (float*)malloc(LLD*sizeof(float));
	h_CO7 = (float*)malloc(LL7*sizeof(float));
	if (h_out == NULL || h_COB == NULL || h_COD == NULL || h_CO7 == NULL) {fprintf(stderr,"Could not allocate host memory\n"); exit(EXIT_FAILURE);}

	err = cudaMalloc((void**)&d_CHUNK, (CS + (2*FILTLENGTH))*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to allocate CHUNK on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMalloc((void**)&d_tempbfft, (CS)*sizeof(float));/////
	if(err != cudaSuccess) {fprintf(stderr, "Failed to allocate tempbfft on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMalloc((void**)&d_tempfftd, (CSD + (2*LLD))*sizeof(float));/////
	if(err != cudaSuccess) {fprintf(stderr, "Failed to allocate tempfftd on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMalloc((void**)&d_diff, (CSD + (2*LL7))*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to allocate diff on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMalloc((void**)&d_del, (CSD + (2*LL7))*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to allocate del on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMalloc((void**)&d_temp7, (CS7 + (LL7))*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to allocate temp7 on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMalloc((void**)&d_out, OS*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to allocate out on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMalloc((void**)&d_COB, FILTLENGTH*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to allocate COB on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMalloc((void**)&d_COD, LLD*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to allocate COD on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMalloc((void**)&d_CO7, LL7*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to allocate CO7 on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMalloc((void**)&d_test, CS*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to allocate test on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}

	/*set up coefficients on device*/
	setup_filter(h_COB, CO, COL, FILTLENGTH);
	err = cudaMemcpy(d_COB, h_COB, FILTLENGTH*sizeof(float), cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {fprintf(stderr, "Failed to copy COB from host to device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	setup_filter(h_COD, CD, LD, LLD);
	err = cudaMemcpy(d_COD, h_COD, LLD*sizeof(float), cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {fprintf(stderr, "Failed to copy COD from host to device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	setup_filter(h_CO7, C7, L7, LL7);
	err = cudaMemcpy(d_CO7, h_CO7, LL7*sizeof(float), cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {fprintf(stderr, "Failed to copy CO7 from host to device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}

	FILE *fout = fopen("output.raw","wb");
	if(fout == NULL) {printf("Can't open output file\n"); exit(EXIT_FAILURE);}

	/*set up temp buffers*/
	err = cudaMemset(d_tempbfft, 0, (CS)*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to set tempbfft on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMemset(d_tempfftd, 0, (CSD + (2*LLD))*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to set tempfftd on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMemset(d_del, 0, (CSD+(2*LL7))*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to set del on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMemset(d_diff, 0, (CSD+(2*LL7))*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to set diff on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMemset(d_temp7, 0, (CS7+LL7)*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to set temp7 on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMemset(d_test, 0, CS*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to set test on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}

	while(semctl(marker,0,GETVAL))
	{
		if(semctl(sfull,semid,GETVAL) == 1)
		{
			sfullset.val = 0;
			if(semctl(sfull,semid,SETVAL,sfullset) == -1) {fprintf(stderr,"semctl %d\n",semid); DIE("semctl");}

			tick++;
			size_t bytes = 0;
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start, 0);

			err = cudaMemcpy(d_CHUNK, s + (semid*smi.chk), CS*sizeof(float), cudaMemcpyHostToDevice);
			if(err != cudaSuccess) {fprintf(stderr, "Failed to copy CHUNK to device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
			semid = (semid + 1) % smi.num;
			err = cudaMemcpy(d_CHUNK + CS, s+(semid*smi.chk), 2*FILTLENGTH*sizeof(float), cudaMemcpyHostToDevice);
			if(err != cudaSuccess) {fprintf(stderr, "Failed to copy CHUNK to device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}

			tempBufSet<<<BD, TD>>>(d_tempfftd, LLD, CSD);
			tempBufSet<<<BD, TD>>>(d_diff, LLD, CSD);
			tempBufSet<<<BD, TD>>>(d_del, LLD, CSD);
			oldTempBufSet<<<B7, T7>>>(d_temp7, LL7, CS7);

			filtbank<<<numBlocks, THREADX, (FILTLENGTH/NUMCHANNELS)*sizeof(float)>>>(d_COB, d_CHUNK, d_tempbfft, SLICESIZE, (FILTLENGTH/NUMCHANNELS), NUMCHANNELS);

			if(cufftExecC2C(plan, (cufftComplex*)d_tempbfft, (cufftComplex*)d_test, CUFFT_INVERSE) != CUFFT_SUCCESS){fprintf(stderr, "ftt didn't work\n"); exit(EXIT_FAILURE);}
			betterMemcpy<<<128,128>>>(d_test + (2*batch*station_num), d_tempfftd + LLD, CSD);

			//err = cudaMemcpy(h_test, d_tempfftd + (2*LLD), (CSD)*sizeof(float), cudaMemcpyDeviceToHost);
			//if(err != cudaSuccess) {fprintf(stderr, "Failed to copy to 'test' on host (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
			//fwrite(h_test, sizeof(float), CSD, fother);

			ddfilt<<<BD, TD, LLD*sizeof(float)>>>(d_COD, d_tempfftd, d_diff, d_del, LLD, SD, DEL, LL7);
			crossmult<<<640, 640>>>(d_diff, d_del, d_temp7, CSD+(2*LL7));
			deciconvo<<<B7, T7, LL7*sizeof(float)>>>(d_CO7, d_temp7, d_out, D7, LL7, T7, S7, 0);

			cudaMemcpy(h_out, d_out, OS*sizeof(float), cudaMemcpyDeviceToHost);
			if(err != cudaSuccess) {fprintf(stderr, "Failed to copy output buffer from device to host (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
			bytes = fwrite(h_out, sizeof(float), OS, fout);

			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			float elapsedTime;
			cudaEventElapsedTime(&elapsedTime, start, stop);
			fprintf(stderr,"Elapsed time: %f\n", elapsedTime);
			cudaEventDestroy(start);
			cudaEventDestroy(stop);

			semptyset.val = 1;
			if(semctl(sempty, old_semid, SETVAL, semptyset) == -1) {fprintf(stderr,"semctl %d\n",old_semid); DIE("semctl");}
			old_semid = semid;
		}
	}
	printf("tick = %d\n", tick);

	fclose(fout);
	cudaFree(d_test);
	cudaFree(d_CHUNK);
	cudaFree(d_tempbfft);
	cudaFree(d_tempfftd);
	cudaFree(d_diff);
	cudaFree(d_del);
	cudaFree(d_temp7);
	cudaFree(d_out);
	cudaFree(d_COB);
	cudaFree(d_COD);
	cudaFree(d_CO7);
	free(h_out);
	free(h_COB);
	free(h_COD);
	free(h_CO7);
}
