#include "CudaRenderer.h"

//==============================================================================================//

REGISTER_OP("CudaRendererGpu")

.Input("vertex_pos: float")
.Input("vertex_color: float")
.Input("texture: float")
.Input("sh_coeff: float")
.Input("target_image: float")
.Input("extrinsics: float")
.Input("intrinsics: float")

.Output("barycentric_buffer: float")
.Output("face_buffer: int32")
.Output("render_buffer: float")
.Output("vertex_normal: float")
.Output("normal_map: float")

.Attr("faces: list(int)")
.Attr("texture_coordinates: list(float)")
.Attr("number_of_vertices: int")
.Attr("number_of_cameras: int")
.Attr("render_resolution_u: int = 512")
.Attr("render_resolution_v: int = 512")
.Attr("albedo_mode: string")
.Attr("shading_mode: string")
.Attr("image_filter_size: int = 2")
.Attr("texture_filter_size: int = 2")
.Attr("compute_normal_map: string");

//==============================================================================================//

CudaRenderer::CudaRenderer(OpKernelConstruction* context)
	: 
	OpKernel(context) 
{
	std::vector<int> faces;
	OP_REQUIRES_OK(context, context->GetAttr("faces", &faces));

	std::vector<float> textureCoordinates;
	OP_REQUIRES_OK(context, context->GetAttr("texture_coordinates", &textureCoordinates));

	OP_REQUIRES_OK(context, context->GetAttr("number_of_vertices", &numberOfPoints));
	OP_REQUIRES(context, numberOfPoints > 0, errors::InvalidArgument("number_of_vertices not set!", numberOfPoints));

	OP_REQUIRES_OK(context, context->GetAttr("number_of_cameras", &numberOfCameras));
	OP_REQUIRES(context, numberOfCameras > 0, errors::InvalidArgument("number_of_cameras not set!", numberOfCameras));

	OP_REQUIRES_OK(context, context->GetAttr("render_resolution_u", &renderResolutionU));
	OP_REQUIRES(context, renderResolutionU > 0, errors::InvalidArgument("render_resolution_u not set!", renderResolutionU));

	OP_REQUIRES_OK(context, context->GetAttr("render_resolution_v", &renderResolutionV));
	OP_REQUIRES(context, renderResolutionV > 0, errors::InvalidArgument("render_resolution_v not set!", renderResolutionV));

	OP_REQUIRES_OK(context, context->GetAttr("albedo_mode", &albedoMode));
	if (albedoMode != "vertexColor" && albedoMode != "textured" && albedoMode != "normal"  && albedoMode != "foregroundMask" && albedoMode != "lighting" && albedoMode != "depth" && albedoMode != "position" && albedoMode != "uv")
	{
		std::cout << "INVALID ALBEDO MODE" << std::endl;
		return;
	}

	OP_REQUIRES_OK(context, context->GetAttr("shading_mode", &shadingMode));
	if (shadingMode != "shaded" && shadingMode != "shadeless")
	{
		std::cout << "INVALID SHADING MODE" << std::endl;
		return;
	}
	if (albedoMode == "normal" || albedoMode == "foregroundMask" || albedoMode == "lighting" || albedoMode == "depth" || albedoMode == "position" || albedoMode == "uv")
	{
		std::cout << "Automatically chose shading mode: shadeless" << std::endl;
		shadingMode = "shadeless";
	}

	
	OP_REQUIRES_OK(context, context->GetAttr("compute_normal_map", &computeNormal));


	if (computeNormal != "normal" && computeNormal != "position" && computeNormal != "none" && computeNormal != "face" )
	{
		std::cout << "INVALID Normal MODE" << std::endl;
		return;
	}

	//---CONSOLE OUTPUT---

	std::cout << std::endl;
	std::cout << "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||" << std::endl;
	std::cout << std::endl;

	/////////////////////////////////////////
	/////////////////////////////////////////

	std::cout << "OPERATOR: CudaRenderer" << std::endl;

	/////////////////////////////////////////
	/////////////////////////////////////////

	//render mode
	if (albedoMode == "vertexColor")
	{
		std::cout << "Albedo mode: vertexColor" << std::endl;
	}
	else if (albedoMode == "textured")
	{
		std::cout << "Albedo mode: textured" << std::endl;
	}
	else if (albedoMode == "normal")
	{
		std::cout << "Albedo mode: normal (note that gradients are zero now)" << std::endl;
	}
	else if (albedoMode == "lighting")
	{
		std::cout << "Albedo mode: lighting (note that gradients are zero now)" << std::endl;
	}
	else if (albedoMode == "foregroundMask")
	{
		std::cout << "Foreground mask mode: automatically choose shadeless" << std::endl;
	}
	else if (albedoMode == "depth")
	{
		std::cout << "Depth mode: automatically choose shadeless" << std::endl;
	}
	else if (albedoMode == "position")
	{
		std::cout << "Position mode: automatically choose shadeless" << std::endl;
	}
	else if (albedoMode == "uv")
	{
		std::cout << "UV mode: automatically choose shadeless" << std::endl;
	}
	if (computeNormal=="normal")
	{std::cout << "Compute Normal3 : " << computeNormal << std::endl;
	}
	else if (computeNormal=="position")
	{std::cout << "Compute Positions : " << computeNormal << std::endl;
	}
	else if (computeNormal=="face")
	{std::cout << "Compute face : " << computeNormal << std::endl;
	}
	else if (computeNormal=="none")
	{std::cout << "Don't Compute Normals " << computeNormal << std::endl;
	}
	/////////////////////////////////////////
	/////////////////////////////////////////

	//shading mode
	if (shadingMode == "shaded")
	{
		std::cout << "Shading mode: shaded" << std::endl;
	}
	else if (shadingMode == "shadeless")
	{
		std::cout << "Shading mode: shadeless" << std::endl;
	}

	/////////////////////////////////////////
	/////////////////////////////////////////

	//number of cameras 
	std::cout << "Number of cameras: " << std::to_string(numberOfCameras) << std::endl;

	/////////////////////////////////////////
	/////////////////////////////////////////

	//render resolution
	std::cout << "Resolution: " << std::to_string(renderResolutionU) << " x " << std::to_string(renderResolutionV) << std::endl;

	/////////////////////////////////////////
	/////////////////////////////////////////

	//number of vertices 
	std::cout << "Number of vertices: " << std::to_string(numberOfPoints) << std::endl;

	std::cout << std::endl;
	std::cout << "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||" << std::endl;
	std::cout << std::endl;

	cudaBasedRasterization = new CUDABasedRasterization(faces, textureCoordinates, numberOfPoints, numberOfCameras, renderResolutionU, renderResolutionV, albedoMode, shadingMode, computeNormal);
}

//==============================================================================================//

void CudaRenderer::setupInputOutputTensorPointers(OpKernelContext* context)
{
	//---INPUT---

	//[0]
	//Grab the 3D vertex position
	const Tensor& inputTensorVertexPos = context->input(0);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorVertexPosFlat = inputTensorVertexPos.flat_inner_dims<float, 1>();
	d_inputVertexPos = inputTensorVertexPosFlat.data();

	//[1]
	//Grab the vertex color
	const Tensor& inputTensorVertexColor = context->input(1);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorVertexColorFlat = inputTensorVertexColor.flat_inner_dims<float, 1>();
	d_inputVertexColor = inputTensorVertexColorFlat.data();

	//[2]
	//Grab the texture
	const Tensor& inputTensorTexture = context->input(2);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorTextureFlat = inputTensorTexture.flat_inner_dims<float, 1>();
	d_inputTexture = inputTensorTextureFlat.data();

	//[3]
	//Grab the sh coeffs 
	const Tensor& inputTensorSHCoeff = context->input(3);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorSHCoeffFlat = inputTensorSHCoeff.flat_inner_dims<float, 1>();
	d_inputSHCoeff = inputTensorSHCoeffFlat.data();

	//[4]
	//Grab the target image
	const Tensor& inputTargetImage = context->input(4);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTargetImageFlat = inputTargetImage.flat_inner_dims<float, 1>();
	d_inputTargetImage = inputTargetImageFlat.data();

	//[5]
	//Grab the extrinsics
	const Tensor& inputExtrinsicsTensor = context->input(5);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputExtrinsicsTensorFlat = inputExtrinsicsTensor.flat_inner_dims<float, 1>();
	d_inputExtrinsics = inputExtrinsicsTensorFlat.data();

	//[6]
	//Grab the intrinsics
	const Tensor& inputIntrinsicsTensor = context->input(6);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputIntrinsicsTensorFlat = inputIntrinsicsTensor.flat_inner_dims<float, 1>();
	d_inputIntrinsics = inputIntrinsicsTensorFlat.data();

	//---MISC---

	numberOfBatches      = inputTensorTexture.dim_size(0);
	textureResolutionV	 = inputTensorTexture.dim_size(1);
	textureResolutionU   = inputTensorTexture.dim_size(2);

	//---OUTPUT---

	//determine the output dimensions

	std::vector<tensorflow::int64> channel1Dim;
	channel1Dim.push_back(numberOfBatches);
	channel1Dim.push_back(numberOfCameras);
	channel1Dim.push_back(renderResolutionV);
	channel1Dim.push_back(renderResolutionU);
	tensorflow::gtl::ArraySlice<tensorflow::int64> channel1DimSize(channel1Dim);

	std::vector<tensorflow::int64> channel2Dim;
	channel2Dim.push_back(numberOfBatches);
	channel2Dim.push_back(numberOfCameras);
	channel2Dim.push_back(renderResolutionV);
	channel2Dim.push_back(renderResolutionU);
	channel2Dim.push_back(2);
	tensorflow::gtl::ArraySlice<tensorflow::int64> channel2DimSize(channel2Dim);

	std::vector<tensorflow::int64> channel3Dim;
	channel3Dim.push_back(numberOfBatches);
	channel3Dim.push_back(numberOfCameras);
	channel3Dim.push_back(renderResolutionV);
	channel3Dim.push_back(renderResolutionU);
	channel3Dim.push_back(3);
	tensorflow::gtl::ArraySlice<tensorflow::int64> channel3DimSize(channel3Dim);

	std::vector<tensorflow::int64> vertexNormalDim;
	vertexNormalDim.push_back(numberOfBatches);
	vertexNormalDim.push_back(numberOfCameras);
	vertexNormalDim.push_back(numberOfPoints);
	vertexNormalDim.push_back(3);
	tensorflow::gtl::ArraySlice<tensorflow::int64> vertexNormalDimSize(vertexNormalDim);

	std::vector<tensorflow::int64> vertexNormalSingleDim;
	vertexNormalSingleDim.push_back(numberOfBatches);
	vertexNormalSingleDim.push_back(textureResolutionV);
	vertexNormalSingleDim.push_back(textureResolutionU);
	vertexNormalSingleDim.push_back(3);
	tensorflow::gtl::ArraySlice<tensorflow::int64> vertexNormalSingleDimSize(vertexNormalSingleDim);

	//[0]
	//barycentric
	tensorflow::Tensor* outputTensorBarycentric;
	OP_REQUIRES_OK(context, context->allocate_output(0, tensorflow::TensorShape(channel2DimSize), &outputTensorBarycentric));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorBarycentricFlat = outputTensorBarycentric->flat<float>();
	d_outputBarycentricCoordinatesBuffer = outputTensorBarycentricFlat.data();

	//[1]
	//face
	tensorflow::Tensor* outputTensorFace;
	OP_REQUIRES_OK(context, context->allocate_output(1, tensorflow::TensorShape(channel1DimSize), &outputTensorFace));
	Eigen::TensorMap<Eigen::Tensor<int, 1, 1, Eigen::DenseIndex>, 16> outputTensorFaceFlat = outputTensorFace->flat<int>();
	d_outputFaceIDBuffer = outputTensorFaceFlat.data();

	//[2]
	//render
	tensorflow::Tensor* outputTensorRender;
	OP_REQUIRES_OK(context, context->allocate_output(2, tensorflow::TensorShape(channel3DimSize), &outputTensorRender));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorRenderFlat = outputTensorRender->flat<float>();
	d_outputRenderBuffer = outputTensorRenderFlat.data();

	//[3]
	//vertex normal
	tensorflow::Tensor* outputTensorVertexNormal;
	OP_REQUIRES_OK(context, context->allocate_output(3, tensorflow::TensorShape(vertexNormalDimSize), &outputTensorVertexNormal));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorVertexNormalFlat = outputTensorVertexNormal->flat<float>();
	d_outputVertexNormal = outputTensorVertexNormalFlat.data();

	//[4]
	//target
	tensorflow::Tensor* outputTensorNormalMap;
	OP_REQUIRES_OK(context, context->allocate_output(4, tensorflow::TensorShape(vertexNormalSingleDimSize), &outputTensorNormalMap));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorNormalMapFlat = outputTensorNormalMap->flat<float>();
	d_outputNormalMap = outputTensorNormalMapFlat.data();
}

//==============================================================================================//

void CudaRenderer::Compute(OpKernelContext* context)
{
    //setup the input and output pointers of the tensor because they change from compute to compute call
    setupInputOutputTensorPointers(context);

    //set input
    cudaBasedRasterization->setTextureWidth(textureResolutionU);
    cudaBasedRasterization->setTextureHeight(textureResolutionV);

    for (int b = 0; b < numberOfBatches; b++)
    {
        //set input
        cudaBasedRasterization->set_D_vertices(			(float3*)   d_inputVertexPos						+ b * numberOfPoints );
        cudaBasedRasterization->set_D_vertexColors(		(float3*)	d_inputVertexColor						+ b * numberOfPoints );
        cudaBasedRasterization->set_D_textureMap(					d_inputTexture							+ b * textureResolutionV * textureResolutionU * 3);
        cudaBasedRasterization->set_D_shCoeff(						d_inputSHCoeff							+ b * numberOfCameras * 27);
        cudaBasedRasterization->set_D_extrinsics(					d_inputExtrinsics						+ b * numberOfCameras * 12);
        cudaBasedRasterization->set_D_intrinsics(					d_inputIntrinsics						+ b * numberOfCameras * 9);

        //set output
        cudaBasedRasterization->set_D_barycentricCoordinatesBuffer(	d_outputBarycentricCoordinatesBuffer	+ b * numberOfCameras * renderResolutionV * renderResolutionU * 2);
        cudaBasedRasterization->set_D_faceIDBuffer(					d_outputFaceIDBuffer					+ b * numberOfCameras * renderResolutionV * renderResolutionU);
        cudaBasedRasterization->set_D_renderBuffer(					d_outputRenderBuffer					+ b * numberOfCameras * renderResolutionV * renderResolutionU * 3);
        cudaBasedRasterization->set_D_vertexNormal(		(float3*)	d_outputVertexNormal					+ b * numberOfCameras * numberOfPoints );
        cudaBasedRasterization->set_D_normalMap(		(float3*)	d_outputNormalMap						+ b * textureResolutionU * textureResolutionV);

        //render
        cudaBasedRasterization->renderBuffers();
    }
}

//==============================================================================================//

REGISTER_KERNEL_BUILDER(Name("CudaRendererGpu").Device(DEVICE_GPU), CudaRenderer);
