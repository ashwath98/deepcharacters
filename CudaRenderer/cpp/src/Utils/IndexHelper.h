//==============================================================================================//
// Classname:
//      Index helper util
//
//==============================================================================================//
// Description:
//      Helper for basic indexing going from kernel ids to multidimensional indices
//
//==============================================================================================//

#pragma once 

//==============================================================================================//

#include <cutil_inline.h>
#include <cutil_math.h>

//==============================================================================================//

__inline__ __device__ int2 index1DTo2D(int size0, int size1, int index1D)
{
	int2 index2D = make_int2(0, 0);

	index2D.y = (index1D % (size1)) ;
	index2D.x = ((index1D - index2D.y) / (size1));

	if (index1D < 0)
	{
		printf("Error negative 1D index \n");
		index2D = make_int2(-1, -1);
	}
	else if (index1D >= size0 * size1)
	{
		printf("Error 1D index out of range \n");
		index2D = make_int2(-1, -1);
	}

	return index2D;
}

//==============================================================================================//

__inline__ __device__ int3 index1DTo3D(int size0, int size1, int size2, int index1D)
{
	int3 index3D = make_int3(0, 0, 0);

	index3D.z = (index1D % (size1*size2)) % size2;
	index3D.y = ((index1D - index3D.z) % (size1*size2)) / size2;
	index3D.x = (index1D - index3D.y * size2 - index3D.z) / (size1 * size2);

	if (index1D < 0)
	{
		printf("Error negative 1D index \n");
		index3D = make_int3(-1, -1, -1);
	}
	else if (index1D >= size0 * size1 * size2)
	{
		printf("Error 1D index out of range \n");
		index3D = make_int3(-1, -1, -1);
	}

	return index3D;
}

//==============================================================================================//

__inline__ __device__ int4 index1DTo4D(int size0, int size1, int size2,int size3, int index1D)
{
	int4 index4D = make_int4(0, 0, 0, 0 );

	index4D.w = (( index1D                                                                % (size1 * size2 * size3)) % ( size2 * size3)) % size3;
	index4D.z = (((index1D - index4D.w)                                                   % (size1 * size2 * size3)) % ( size2 * size3)) / size3;
	index4D.y = (((index1D - index4D.w - index4D.z * size3)                               % (size1 * size2 * size3))) / (size2 * size3);
	index4D.x = (((index1D - index4D.w - index4D.z * size3 - index4D.y * size2 * size3))) / (size1 * size2 * size3);

	if (index1D < 0)
	{
		printf("Error negative 1D index \n");
		index4D = make_int4(-1, -1, -1, -1);
	}
	else if (index1D >= size0 * size1 * size2 * size3)
	{
		printf("Error 1D index out of range \n");
		index4D = make_int4(-1, -1, -1, -1);
	}

	return index4D;
}

//==============================================================================================//

__inline__ __device__ int index5DTo1D(int size0, int size1, int size2, int size3, int size4, int id0, int id1, int id2, int id3, int id4)
{
	int index = id0 * size1 * size2* size3 * size4 + id1 * size2 * size3  * size4 + id2 * size3  * size4 + id3 * size4 + id4;

	if (index < 0)
		index = -1;
	if (index >= size0*size1*size2*size3*size4)
		index = -1;
	if (id0 < 0 || id0 >= size0)
		index = -1;
	if (id1 < 0 || id1 >= size1)
		index = -1;
	if (id2 < 0 || id2 >= size2)
		index = -1;
	if (id3 < 0 || id3 >= size3)
		index = -1;
	if (id4 < 0 || id4 >= size4)
		index = -1;

	return index;
}

//==============================================================================================//

__inline__ __device__ int index4DTo1D(int size0, int size1, int size2, int size3, int id0, int id1, int id2, int id3)
{
	int index = id0 * size1 * size2* size3  + id1 * size2 * size3 + id2 * size3 + id3 ;

	if (index < 0)
		index = -1;
	if(index >= size0*size1*size2*size3)
		index = -1;
	if (id0 < 0 || id0 >= size0)
		index = -1;
	if (id1 < 0 || id1 >= size1)
		index = -1;
	if (id2 < 0 || id2 >= size2)
		index = -1;
	if (id3 < 0 || id3 >= size3)
		index = -1;

	return index;
}

//==============================================================================================//

__inline__ __device__ int index3DTo1D(int size0, int size1, int size2, int id0, int id1, int id2)
{
	int index = id0 * size1 * size2 + id1 * size2 + id2;

	if (index < 0)
		index = -1;
	if (index >= size0*size1*size2)
		index = -1;
	if (id0 < 0 || id0 >= size0)
		index = -1;
	if (id1 < 0 || id1 >= size1)
		index = -1;
	if (id2 < 0 || id2 >= size2)
		index = -1;

	return index;
}

//==============================================================================================//

__inline__ __device__ int index2DTo1D(int size0, int size1, int id0, int id1)
{
	int index = id0 * size1  + id1 ;

	if (index < 0)
		index = -1;
	if (index >= size0*size1)
		index = -1;
	if (id0 < 0 || id0 >= size0)
		index = -1;
	if (id1 < 0 || id1 >= size1)
		index = -1;

	return index;
}