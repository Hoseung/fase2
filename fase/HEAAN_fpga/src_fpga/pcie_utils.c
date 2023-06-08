/*
 * This file is part of the Xilinx DMA IP Core driver tools for Linux
 *
 * Copyright (c) 2016-present,  Xilinx, Inc.
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <unistd.h>
#include <time.h>
#include <errno.h>
//#include <sys/types.h>

#include <assert.h>
#include <fcntl.h>
#include <getopt.h>
#include <string.h>
#include <time.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#define MASK_64_0          0x00000000000000FF
#define MASK_64_1          0x000000000000FF00
#define MASK_64_2          0x0000000000FF0000
#define MASK_64_3          0x00000000FF000000
#define MASK_64_4          0x000000FF00000000
#define MASK_64_5          0x0000FF0000000000
#define MASK_64_6          0x00FF000000000000
#define MASK_64_7          0xFF00000000000000

#define RW_MAX_SIZE	0x7ffff000
#define MAP_SIZE (8*1024UL) //8k byte

int verbose = 0;
int debug_value = 0;

uint64_t getopt_integer(char *optarg)
{
	int rc;
	uint64_t value;

	rc = sscanf(optarg, "0x%lx", &value);
	if (rc <= 0)
		rc = sscanf(optarg, "%lu", &value);
	//printf("sscanf() = %d, value = 0x%lx\n", rc, value);

	return value;
}

unsigned int reg_read( char *device, long addr){
    int fd;
    void *map_base, *virt_addr;
    uint32_t read_result;

    if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 

    if ((fd = open(device, O_RDWR | O_SYNC)) == -1) {
            fprintf(stderr, "Error at line %d, function(%s) (%d) [%s]\n", __LINE__, __FUNCTION__, errno, strerror(errno)); 
        goto out; 
    }
    if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
    if (verbose) fprintf(stdout, "** Info: device %s opened.\n", device);

    /* map one page */
    map_base = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map_base == (void *)-1) {
            fprintf(stderr, "Error at line %d, function(%s) (%d) [%s]\n", __LINE__, __FUNCTION__, errno, strerror(errno)); 
        goto out; 
    }
    if (debug_value) {
        fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
        fprintf(stdout, "## Debug: Memory mapped at address %p.\n", map_base);
    }

    virt_addr = map_base + addr;
    read_result = *((uint32_t *) virt_addr);
    if (debug_value) {
        fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
        fprintf(stdout, "## Debug: read value of virtual addr(0x%p, addr:0x%lx) is 0x%x.\n", virt_addr, addr, read_result);
    }

    if (munmap(map_base, MAP_SIZE) == -1)
            fprintf(stderr, "Error at line %d, file %s (%d) [%s]\n", __LINE__, __FILE__, errno, strerror(errno)); 
out:
    close(fd);

    return read_result;
}

int reg_write( char *device, long addr, unsigned int data){
    int fd;
    void *map_base, *virt_addr;
    uint32_t read_result;

    if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 

    if ((fd = open(device, O_RDWR | O_SYNC)) == -1) {
            fprintf(stderr, "Error at line %d, file %s (%d) [%s]\n", __LINE__, __FILE__, errno, strerror(errno)); 
        goto out; 
    }
    if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
    if (verbose) fprintf(stdout, "** Info: device %s opened.\n", device);

    /* map one page */
    map_base = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map_base == (void *)-1) {
            fprintf(stderr, "Error at line %d, file %s (%d) [%s]\n", __LINE__, __FILE__, errno, strerror(errno)); 
        goto out; 
    }
    if (debug_value) {
        fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
        fprintf(stdout, "## Debug: Memory mapped at address %p.\n", map_base);
    }

    virt_addr = map_base + addr;
    *((uint32_t *) virt_addr) = data;
    if (debug_value) {
        fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
        fprintf(stdout, "## Debug: write value of virtual addr(0x%p, addr:0x%lx) is 0x%x.\n", virt_addr, addr, data);
    }

    if (munmap(map_base, MAP_SIZE) == -1) 
            fprintf(stderr, "Error at line %d, file %s (%d) [%s]\n", __LINE__, __FILE__, errno, strerror(errno)); 

out: 
    close(fd);

    return 0;
}

ssize_t read_to_buffer(char *fname, int fd, char *buffer, uint64_t size,
			uint64_t base)
{
	ssize_t rc;
	uint64_t count = 0;
	char *buf = buffer;
	off_t offset = base;

	if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
	while (count < size) {
		uint64_t bytes = size - count;

		if (bytes > RW_MAX_SIZE)
			bytes = RW_MAX_SIZE;

		if (offset) {
			if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
			rc = lseek(fd, offset, SEEK_SET);
			if (rc != offset) {
				fprintf(stderr, "%s, seek off 0x%lx != 0x%lx.\n", fname, rc, offset);
				perror("seek file");
				return -EIO;
			}
		}

		/* read data from file into memory buffer */
		if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
		rc = read(fd, buf, bytes);
		if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
		if (rc != bytes) {
			fprintf(stderr, "%s, R off 0x%lx, 0x%lx != 0x%lx.\n", fname, count, rc, bytes);
			perror("read file");
			return -EIO;
		}

		count += bytes;
		buf += bytes;
		offset += bytes;
		if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
		if (debug_value) fprintf(stdout, "## Debug: count(%ld), size(%ld)\n",count, size); 
	}	 

	if (count != size) {
		fprintf(stderr, "%s, R failed 0x%lx != 0x%lx.\n", fname, count, size);
		return -EIO;
	}
	return count;
}

ssize_t write_from_buffer(char *fname, int fd, char *buffer, uint64_t size,
			uint64_t base)
{
	ssize_t rc;
	uint64_t count = 0;
	char *buf = buffer;
	off_t offset = base;

	if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
	while (count < size) {
		uint64_t bytes = size - count;

		if (bytes > RW_MAX_SIZE)
			bytes = RW_MAX_SIZE;

	        if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
		if (offset) {
	                if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
			rc = lseek(fd, offset, SEEK_SET);
			if (rc != offset) {
				fprintf(stderr, "%s, seek off 0x%lx != 0x%lx.\n",
					fname, rc, offset);
				perror("seek file");
				return -EIO;
			}
		}

		/* write data to file from memory buffer */
	        if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
		rc = write(fd, buf, bytes);
	        if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
		if (rc != bytes) {
			fprintf(stderr, "%s, W off 0x%lx, 0x%lx != 0x%lx.\n",
				fname, offset, rc, bytes);
				perror("write file");
			return -EIO;
		}

		count += bytes;
		buf += bytes;
		offset += bytes;
		if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
		if (debug_value) fprintf(stdout, "## Debug: count(%ld), size(%ld)\n",count, size); 
	}	 

	if (count != size) {
		fprintf(stderr, "%s, R failed 0x%lx != 0x%lx.\n", fname, count, size);
		return -EIO;
	}
	return count;
}

ssize_t write_one_from_buffer(char *fname, int fd, char *buffer, uint64_t base)
{
	ssize_t rc;
	uint64_t count = 0;
	char *buf = buffer;
	off_t offset = base;

	if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
	uint64_t bytes = 4;

	if (offset) {
	        if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
		rc = lseek(fd, offset, SEEK_SET);
		if (rc != offset) {
			fprintf(stderr, "%s, seek off 0x%lx != 0x%lx.\n", fname, rc, offset);
			perror("seek file");
			return -EIO;
		}
	}

	/* write data to file from memory buffer */
	if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
	rc = write(fd, buf, 4);
	if (debug_value) fprintf(stdout, "## Debug: rc  = %ld\n", rc);
	if (rc != 4) {
		fprintf(stderr, "%s, W off 0x%lx, 0x%lx != 0x%lx.\n", fname, offset, rc, bytes);
		perror("write file");
		return -EIO;
	}


	return 4;
}


/* Subtract timespec t2 from t1
 *
 * Both t1 and t2 must already be normalized
 * i.e. 0 <= nsec < 1000000000
 */
static int timespec_check(struct timespec *t)
{
	if ((t->tv_nsec < 0) || (t->tv_nsec >= 1000000000))
		return -1;
	return 0;

}

void timespec_sub(struct timespec *t1, struct timespec *t2)
{
	if (timespec_check(t1) < 0) {
		fprintf(stderr, "invalid time #1: %lld.%.9ld.\n",
			(long long)t1->tv_sec, t1->tv_nsec);
		return;
	}
	if (timespec_check(t2) < 0) {
		fprintf(stderr, "invalid time #2: %lld.%.9ld.\n",
			(long long)t2->tv_sec, t2->tv_nsec);
		return;
	}
	t1->tv_sec -= t2->tv_sec;
	t1->tv_nsec -= t2->tv_nsec;
	if (t1->tv_nsec >= 1000000000) {
		t1->tv_sec++;
		t1->tv_nsec -= 1000000000;
	} else if (t1->tv_nsec < 0) {
		t1->tv_sec--;
		t1->tv_nsec += 1000000000;
	}
}

int dma_to_device_from_file(char *devname, uint64_t addr, uint64_t size, uint64_t offset, uint64_t count, char *infname, char *ofname)
{
	uint64_t i;
	ssize_t rc;
	char *buffer = NULL;
	char *allocated = NULL;
	struct timespec ts_start, ts_end;
	int infile_fd = -1;
	int outfile_fd = -1;
	int fpga_fd = open(devname, O_RDWR);
	long total_time = 0;
	float result;
	float avg_time = 0;

	if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 

	if (fpga_fd < 0) {
		fprintf(stderr, "unable to open device %s, %d.\n",
			devname, fpga_fd);
		perror("open device");
		return -EINVAL;
	}

	if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
	if (infname) {
		infile_fd = open(infname, O_RDONLY);
		if (infile_fd < 0) {
			fprintf(stderr, "unable to open input file %s, %d.\n",
				infname, infile_fd);
			perror("open input file");
			rc = -EINVAL;
			goto out;
		}
	}

	if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
	if (ofname) {
		outfile_fd =
		    open(ofname, O_RDWR | O_CREAT | O_TRUNC | O_SYNC,
			 0666);
		if (outfile_fd < 0) {
			fprintf(stderr, "unable to open output file %s, %d.\n",
				ofname, outfile_fd);
			perror("open output file");
			rc = -EINVAL;
			goto out;
		}
	}

	if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
	posix_memalign((void **)&allocated, 4096 /*alignment */ , size + 4096);
	if (!allocated) {
		fprintf(stderr, "OOM %lu.\n", size + 4096);
		rc = -ENOMEM;
		goto out;
	}
	if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
	buffer = allocated + offset;
	if (verbose) fprintf(stdout, "host buffer 0x%lx = %p\n", size + 4096, buffer); 

	if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
	if (infile_fd >= 0) {
	        //fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
		rc = read_to_buffer(infname, infile_fd, buffer, size, 0);
		if (rc < 0)
			goto out;
	}

	if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
	for (i = 0; i < count; i++) {
		/* write buffer to AXI MM address using SGDMA */
	        if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
		rc = clock_gettime(CLOCK_MONOTONIC, &ts_start);

	        if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
		rc = write_from_buffer(devname, fpga_fd, buffer, size, addr);
		if (rc < 0) goto out;

	        if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
		rc = clock_gettime(CLOCK_MONOTONIC, &ts_end);
		/* subtract the start time from the end time */
	        if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
		timespec_sub(&ts_end, &ts_start);
		total_time += ts_end.tv_nsec;
	        if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
		/* a bit less accurate but side-effects are accounted for */
		if (verbose) fprintf(stdout, "#%lu: CLOCK_MONOTONIC %ld.%09ld sec. write %ld bytes\n", i, ts_end.tv_sec, ts_end.tv_nsec, size); 
			
	        if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
		if (outfile_fd >= 0) {
			rc = write_from_buffer(ofname, outfile_fd, buffer,
						 size, i * size);
			if (rc < 0)
				goto out;
		}
	}
	avg_time = (float)total_time/(float)count;
	result = ((float)size)*1000/avg_time;
	if (verbose) printf("** Avg time device %s, total time %ld nsec, avg_time = %f, size = %lu, BW = %f \n", devname, total_time, avg_time, size, result);

	if (verbose) printf("** Average BW = %lu, %f\n",size, result);
	rc = 0;

out:
	if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
	close(fpga_fd);
	if (infile_fd >= 0)
		close(infile_fd);
	if (outfile_fd >= 0)
		close(outfile_fd);
	free(allocated);

	return rc;
}

//int dma_to_device(char *devname, uint64_t addr, uint64_t size, uint64_t offset, uint64_t count, char *buffer)
int dma_to_device(char *devname, uint64_t addr, uint64_t size, uint64_t offset, uint64_t count, uint64_t *data)
{
	uint64_t i, j;
	uint64_t wdata;
	ssize_t rc;
	char *buffer = NULL;
	char *allocated = NULL;
	struct timespec ts_start, ts_end;
	int fpga_fd = open(devname, O_RDWR);
	long total_time = 0;
	float result;
	float avg_time = 0;

	if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 

	if (fpga_fd < 0) {
		fprintf(stderr, "unable to open device %s, %d.\n", devname, fpga_fd);
		perror("open device");
		return -EINVAL;
	}

	if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
	posix_memalign((void **)&allocated, 4096 /*alignment */ , size + 4096);
	if (!allocated) {
		fprintf(stderr, "OOM %lu.\n", size + 4096);
		rc = -ENOMEM;
		goto out;
	}
	if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
	buffer = allocated + offset;
	if (verbose) fprintf(stdout, "host buffer 0x%lx = %p\n", size + 4096, buffer); 

	// copy data to buffer
	if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 

	for(i=0, j=0; i < (size/8); i++) {
		wdata = data[i];

		buffer[j++] = (MASK_64_0 & wdata); 
		buffer[j++] = ((MASK_64_1 & wdata) >> 8); 
		buffer[j++] = ((MASK_64_2 & wdata) >> 16); 
		buffer[j++] = ((MASK_64_3 & wdata) >> 24); 
		buffer[j++] = ((MASK_64_4 & wdata) >> 32); 
		buffer[j++] = ((MASK_64_5 & wdata) >> 40); 
		buffer[j++] = ((MASK_64_6 & wdata) >> 48); 
		buffer[j++] = ((MASK_64_7 & wdata) >> 56); 
	}

	if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
	for (i = 0; i < count; i++) {
		/* write buffer to AXI MM address using SGDMA */
	        if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
		rc = clock_gettime(CLOCK_MONOTONIC, &ts_start);

	        if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
		rc = write_from_buffer(devname, fpga_fd, buffer, size, addr);
		if (rc < 0) goto out;

	        if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
		rc = clock_gettime(CLOCK_MONOTONIC, &ts_end);
		/* subtract the start time from the end time */
	        if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
		timespec_sub(&ts_end, &ts_start);
		total_time += ts_end.tv_nsec;
	        if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
		/* a bit less accurate but side-effects are accounted for */
		if (verbose) fprintf(stdout, "#%lu: CLOCK_MONOTONIC %ld.%09ld sec. write %ld bytes\n", i, ts_end.tv_sec, ts_end.tv_nsec, size); 
			
	        if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
	}
	avg_time = (float)total_time/(float)count;
	result = ((float)size)*1000/avg_time;
	if (verbose) printf("** Avg time device %s, total time %ld nsec, avg_time = %f, size = %lu, BW = %f \n", devname, total_time, avg_time, size, result);

	if (verbose) printf("** Average BW = %lu, %f\n",size, result);
	rc = 0;

out:
	if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
	close(fpga_fd);
	free(allocated);

	return rc;
}

int dma_from_device_to_file(char *devname, uint64_t addr, uint64_t size, uint64_t offset, uint64_t count, char *ofname)
{
	ssize_t rc;
	uint64_t i;
	char *buffer = NULL;
	char *allocated = NULL;
	struct timespec ts_start, ts_end;
	int out_fd = -1;
	int fpga_fd = open(devname, O_RDWR | O_NONBLOCK);
	long total_time = 0;
	float result;
	float avg_time = 0;

	if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
	if (fpga_fd < 0) {
                fprintf(stderr, "unable to open device %s, %d.\n",
                        devname, fpga_fd);
		perror("open device");
                return -EINVAL;
        }

	if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
	/* create file to write data to */
	if (ofname) {
		out_fd = open(ofname, O_RDWR | O_CREAT | O_TRUNC | O_SYNC,
				0666);
		if (out_fd < 0) {
                        fprintf(stderr, "unable to open output file %s, %d.\n",
                                ofname, out_fd);
			perror("open output file");
                        rc = -EINVAL;
                        goto out;
                }
	}

	if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
	posix_memalign((void **)&allocated, 4096 /*alignment */ , size + 4096);
	if (!allocated) {
		fprintf(stderr, "OOM %lu.\n", size + 4096);
		rc = -ENOMEM;
		goto out;
	}

	if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
	buffer = allocated + offset;
	if (verbose) fprintf(stdout, "host buffer 0x%lx, %p.\n", size + 4096, buffer);

	for (i = 0; i < count; i++) {
	        if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
		rc = clock_gettime(CLOCK_MONOTONIC, &ts_start);
		/* lseek & read data from AXI MM into buffer using SGDMA */
		rc = read_to_buffer(devname, fpga_fd, buffer, size, addr);
		if (rc < 0) goto out;
	        if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
		clock_gettime(CLOCK_MONOTONIC, &ts_end);

		/* subtract the start time from the end time */
		timespec_sub(&ts_end, &ts_start);
		total_time += ts_end.tv_nsec;
		/* a bit less accurate but side-effects are accounted for */
		if (verbose) fprintf(stdout, "#%lu: CLOCK_MONOTONIC %ld.%09ld sec. read %ld bytes\n", i, ts_end.tv_sec, ts_end.tv_nsec, size);

		/* file argument given? */
		if (out_fd >= 0) {
	                if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
			rc = write_from_buffer(ofname, out_fd, buffer,
					 size, i*size);
			if (rc < 0)
				goto out;
		}
	}
	avg_time = (float)total_time/(float)count;
	result = ((float)size)*1000/avg_time;
	if (verbose) printf("** Avg time device %s, total time %ld nsec, avg_time = %f, size = %lu, BW = %f \n", devname, total_time, avg_time, size, result);

	if (verbose) printf("** Average BW = %lu, %f\n", size, result);
	rc = 0;

out:
	if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
	close(fpga_fd);
	if (out_fd >= 0)
		close(out_fd);
	free(allocated);

	return rc;
}

int dma_from_device(char *devname, uint64_t addr, uint64_t size, uint64_t offset, uint64_t count, uint64_t *data)
{
	ssize_t rc;
	uint64_t i, j, k;
	uint64_t rdata;
	char *buffer = NULL;
	char *allocated = NULL;
	struct timespec ts_start, ts_end;
	int fpga_fd = open(devname, O_RDWR | O_NONBLOCK);
	long total_time = 0;
	float result;
	float avg_time = 0;

	if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
	if (fpga_fd < 0) {
                fprintf(stderr, "unable to open device %s, %d.\n",
                        devname, fpga_fd);
		perror("open device");
                return -EINVAL;
        }

	if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
	posix_memalign((void **)&allocated, 4096 /*alignment */ , size + 4096);
	if (!allocated) {
		fprintf(stderr, "OOM %lu.\n", size + 4096);
		rc = -ENOMEM;
		goto out;
	}

	if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
	buffer = allocated + offset;
	if (verbose) fprintf(stdout, "host buffer 0x%lx, %p.\n", size + 4096, buffer);

	for (i = 0; i < count; i++) {
	        if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
		rc = clock_gettime(CLOCK_MONOTONIC, &ts_start);

		// lseek & read data from AXI MM into buffer using SGDMA 
		rc = read_to_buffer(devname, fpga_fd, buffer, size, addr);
		if (rc < 0) goto out;
	        if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
		clock_gettime(CLOCK_MONOTONIC, &ts_end);

		for(j=0, k=0; j < (size/8); j++) {
			rdata = 0; 

			rdata +=  ((uint64_t)buffer[k++]) & MASK_64_0;
			rdata += (((uint64_t)buffer[k++]) & MASK_64_0) << 8; 
			rdata += (((uint64_t)buffer[k++]) & MASK_64_0) << 16; 
			rdata += (((uint64_t)buffer[k++]) & MASK_64_0) << 24; 
			rdata += (((uint64_t)buffer[k++]) & MASK_64_0) << 32; 
			rdata += (((uint64_t)buffer[k++]) & MASK_64_0) << 40; 
			rdata += (((uint64_t)buffer[k++]) & MASK_64_0) << 48; 
			rdata += (((uint64_t)buffer[k++]) & MASK_64_0) << 56; 

			data[j] = rdata;
		}

		// subtract the start time from the end time 
		timespec_sub(&ts_end, &ts_start);
		total_time += ts_end.tv_nsec;
		// a bit less accurate but side-effects are accounted for 
		if (verbose) fprintf(stdout, "#%lu: CLOCK_MONOTONIC %ld.%09ld sec. read %ld bytes\n", i, ts_end.tv_sec, ts_end.tv_nsec, size);
	}

	
	avg_time = (float)total_time/(float)count;
	result = ((float)size)*1000/avg_time;
	if (verbose) printf("** Avg time device %s, total time %ld nsec, avg_time = %f, size = %lu, BW = %f \n", devname, total_time, avg_time, size, result);

	if (verbose) printf("** Average BW = %lu, %f\n", size, result);
	rc = 0;

out:
	if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
	close(fpga_fd);
	free(allocated);

	return rc;
}

int check_channel() {
    int i;
    int check_value;
    unsigned addr;
    unsigned pcireg_value;
    int c2h_count=0, h2c_count=0;

    // check channel
    fprintf(stdout, "\nCheck PCIe channel\n");
    for(i=0; i<=3; i++) {
        addr = 0x0000 + 0x100 * i;
            pcireg_value = reg_read("/dev/xdma0_control", addr);
            if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
        if (debug_value) fprintf(stdout, "## Debug: h2cChannel %d (0x%08x) is 0x%08x \n", i, addr, pcireg_value);

        check_value = pcireg_value >> 20;
        if (debug_value) fprintf(stdout, "## Debug: check_value is 0x%08x \n", check_value);
        if(check_value == 0x1fc) {
            if (verbose) fprintf(stdout, "** Info: ID of h2cChannel(0x%08x) is 0x%03x \n", addr, check_value);
            h2c_count++;
        }
        if (debug_value) fprintf(stdout, "## Debug: h2c_count %d\n", h2c_count);
    }
    for(i=0; i<=3; i++) {
        addr = 0x1000 + 0x100 * i;
            pcireg_value = reg_read("/dev/xdma0_control", addr);
            if (debug_value) fprintf(stdout, "## Debug: line %d, function(%s)\n", __LINE__, __FUNCTION__); 
        if (debug_value) fprintf(stdout, "## Debug: c2hChannel %d (0x%08x) is 0x%08x \n", i, addr, pcireg_value);

        check_value = pcireg_value >> 20;
        if(check_value == 0x1fc) c2h_count++;
        if (debug_value) fprintf(stdout, "## Debug: c2h_count %d\n", c2h_count);
    }
    printf(" h2c_count %d\n", h2c_count);
    printf(" c2h_count %d\n", c2h_count);

    if((h2c_count == 0) && (c2h_count ==0)) {
        printf("## Error: No PCIe DMA channels wer identified\n");
        return 1;
    }
    else fprintf(stdout, "Checked PCIe channel\n\n");

    return 0;
}	
