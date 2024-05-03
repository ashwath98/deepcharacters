
#include "CudaRendererGrad.h"

//==============================================================================================//

REGISTER_OP("CudaRendererGradGpu")

.Input("render_buffer_grad: float")
.Input("vertex_pos: float")
.Input("vertex_color: float")
.Input("texture: float")
.Input("sh_coeff: float")
.Input("target_image: float")
.Input("vertex_normal: float")
.Input("barycentric_buffer: float")
.Input("face_buffer: int32")
.Input("extrinsics: float")
.Input("intrinsics: float")

.Output("vertex_pos_grad: float")
.Output("vertex_color_grad: float")
.Output("texture_grad: float")
.Output("sh_coeff_grad: float")

.Attr("faces: list(int)")
.Attr("texture_coordinates: list(float)")
.Attr("number_of_vertices: int")
.Attr("number_of_cameras: int")
.Attr("render_resolution_u: int = 512")
.Attr("render_resolution_v: int = 512")
.Attr("albedo_mode: string")
.Attr("shading_mode: string")
.Attr("image_filter_size: int = 2")
.Attr("texture_filter_size: int = 2");

//==============================================================================================//

CudaRendererGrad::CudaRendererGrad(OpKernelConstruction* context)
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
	if (albedoMode == "foregroundMask" || albedoMode == "normal" || albedoMode == "foregroundMask" || albedoMode == "lighting" || albedoMode == "depth" || albedoMode == "position" || albedoMode == "uv")
	{
		std::cout << "Automatically chose shading mode: shadeless" << std::endl;
		shadingMode = "shadeless";
	}

	int imageFilterSize = -1;
	OP_REQUIRES_OK(context, context->GetAttr("image_filter_size", &imageFilterSize));
	if (imageFilterSize <=0)
	{
		std::cout << "INVALID IMAGE FILTER SIZE" << std::endl;
		return;
	}

	int textureFilterSize = -1;
	OP_REQUIRES_OK(context, context->GetAttr("texture_filter_size", &textureFilterSize));
	if (textureFilterSize <= 0)
	{
		std::cout << "INVALID TEXTURE FILTER SIZE" << std::endl;
		return;
	}

	cudaBasedRasterizationGrad = new CUDABasedRasterizationGrad(faces, textureCoordinates, numberOfPoints, numberOfCameras, renderResolutionU, renderResolutionV, albedoMode, shadingMode, imageFilterSize, textureFilterSize);
}

//==============================================================================================//

void CudaRendererGrad::setupInputOutputTensorPointers(OpKernelContext* context)
{
	//---INPUT---

	/////////////
	//INPUT FROM LATER LAYERS
	/////////////

	//[0]
	//Grab the vertec color buffer gradients 
	const Tensor& inputTensorRenderBufferGrad = context->input(0);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorRenderBufferGradFlat = inputTensorRenderBufferGrad.flat_inner_dims<float, 1>();
	d_inputRenderBufferGrad = inputTensorRenderBufferGradFlat.data();

	/////////////
	//INPUT FROM INPUT OF FORWARD
	/////////////

	//[1]
	//Grab the 3D vertex position
	const Tensor& inputTensorVertexPos = context->input(1);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorVertexPosFlat = inputTensorVertexPos.flat_inner_dims<float, 1>();
	d_inputVertexPos = inputTensorVertexPosFlat.data();

	//[2]
	//Grab the vertex color
	const Tensor& inputTensorVertexColor = context->input(2);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorVertexColorFlat = inputTensorVertexColor.flat_inner_dims<float, 1>();
	d_inputVertexColor = inputTensorVertexColorFlat.data();

	//[3]
	//Grab the texture
	const Tensor& inputTensorTexture = context->input(3);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorTextureFlat = inputTensorTexture.flat_inner_dims<float, 1>();
	d_inputTexture = inputTensorTextureFlat.data();

	//[4]
	//Grab the sh coeffs 
	const Tensor& inputTensorSHCoeff = context->input(4);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorSHCoeffFlat = inputTensorSHCoeff.flat_inner_dims<float, 1>();
	d_inputSHCoeff = inputTensorSHCoeffFlat.data();

	//[5]
	//Grab the target image
	const Tensor& inputTensorTargetImage = context->input(5);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorTargetImageFlat = inputTensorTargetImage.flat_inner_dims<float, 1>();
	d_inputTargetImage = inputTensorTargetImageFlat.data();

	/////////////
	//INPUT FROM OUTPUT OF FORWARD
	/////////////

	//[6]
	//Grab the vertex normals 
	const Tensor& inputTensorVertexNormal = context->input(6);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorVertexNormalFlat = inputTensorVertexNormal.flat_inner_dims<float, 1>();
	d_inputVertexNormal = inputTensorVertexNormalFlat.data();

	//[7]
	//Grab the barycentric co-ordinates 
	const Tensor& inputTensorBaryCentricBuffer= context->input(7);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputTensorBaryCentricBufferFlat = inputTensorBaryCentricBuffer.flat_inner_dims<float, 1>();
	d_inputBaryCentricBuffer = inputTensorBaryCentricBufferFlat.data();

	//[8]
	//Grab the face id buffer  
	const Tensor& inputTensorFaceBuffer = context->input(8);
	Eigen::TensorMap<Eigen::Tensor< const int, 1, 1, Eigen::DenseIndex>, 16> inputTensorFaceBufferFlat = inputTensorFaceBuffer.flat_inner_dims<int, 1>();
	d_inputFaceBuffer= inputTensorFaceBufferFlat.data();

	//[9]
	//Grab the extrinsics
	const Tensor& inputExtrinsicsTensor = context->input(9);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputExtrinsicsTensorFlat = inputExtrinsicsTensor.flat_inner_dims<float, 1>();
	d_inputExtrinsics = inputExtrinsicsTensorFlat.data();

	//[10]
	//Grab the intrinsics
	const Tensor& inputIntrinsicsTensor = context->input(10);
	Eigen::TensorMap<Eigen::Tensor< const float, 1, 1, Eigen::DenseIndex>, 16> inputIntrinsicsTensorFlat = inputIntrinsicsTensor.flat_inner_dims<float, 1>();
	d_inputIntrinsics = inputIntrinsicsTensorFlat.data();

	//---MISC---

	numberOfBatches      = inputTensorVertexPos.dim_size(0); 
	textureResolutionV   = inputTensorTexture.dim_size(1);
	textureResolutionU   = inputTensorTexture.dim_size(2);

	//---OUTPUT---

	//determine the output dimensions
	std::vector<tensorflow::int64> vertexDim;
	vertexDim.push_back(numberOfBatches);
	vertexDim.push_back(numberOfPoints);
	vertexDim.push_back(3);
	tensorflow::gtl::ArraySlice<tensorflow::int64> vertexDimSize(vertexDim);

	std::vector<tensorflow::int64> texDim;
	texDim.push_back(numberOfBatches);
	texDim.push_back(textureResolutionV);
	texDim.push_back(textureResolutionU);
	texDim.push_back(3);
	tensorflow::gtl::ArraySlice<tensorflow::int64> texDimSize(texDim);

	std::vector<tensorflow::int64> shDim;
	shDim.push_back(numberOfBatches);
	shDim.push_back(numberOfCameras);
	shDim.push_back(27);
	tensorflow::gtl::ArraySlice<tensorflow::int64> shDimSize(shDim);

	//[0]
	//vertex position gradients
	tensorflow::Tensor* outputTensorVertexPosGrad;
	OP_REQUIRES_OK(context, context->allocate_output(0, tensorflow::TensorShape(vertexDimSize), &outputTensorVertexPosGrad));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorVertexPosGradFlat = outputTensorVertexPosGrad->flat<float>();
	d_outputVertexPosGrad = outputTensorVertexPosGradFlat.data();

	//[1]
	//vertex color gradients
	tensorflow::Tensor* outputTensorVertexColorGrad;
	OP_REQUIRES_OK(context, context->allocate_output(1, tensorflow::TensorShape(vertexDimSize), &outputTensorVertexColorGrad));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorVertexColorGradFlat = outputTensorVertexColorGrad->flat<float>();
	d_outputVertexColorGrad = outputTensorVertexColorGradFlat.data();

	//[2]
	//texture gradients
	tensorflow::Tensor* outputTensorTextureGrad;
	OP_REQUIRES_OK(context, context->allocate_output(2, tensorflow::TensorShape(texDimSize), &outputTensorTextureGrad));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorTextureGradFlat = outputTensorTextureGrad->flat<float>();
	d_outputTextureGrad = outputTensorTextureGradFlat.data();

	//[3]
	//sh coeff gradients
	tensorflow::Tensor* outputTensorSHCoeffGrad;
	OP_REQUIRES_OK(context, context->allocate_output(3, tensorflow::TensorShape(shDimSize), &outputTensorSHCoeffGrad));
	Eigen::TensorMap<Eigen::Tensor<float, 1, 1, Eigen::DenseIndex>, 16> outputTensorSHCoeffGradFlat= outputTensorSHCoeffGrad->flat<float>();
	d_outputSHCoeffGrad = outputTensorSHCoeffGradFlat.data();
}

//==============================================================================================//

void CudaRendererGrad::Compute(OpKernelContext* context)
{
    //setup the input and output pointers of the tensor because they change from compute to compute call
    setupInputOutputTensorPointers(context);

    for (int b = 0; b < numberOfBatches; b++)
    {
        //set input
        cudaBasedRasterizationGrad->setTextureWidth(textureResolutionU);
        cudaBasedRasterizationGrad->setTextureHeight(textureResolutionV);
        cudaBasedRasterizationGrad->set_D_RenderBufferGrad(				(float3*)			d_inputRenderBufferGrad					+ b * numberOfCameras * renderResolutionV * renderResolutionU);
        cudaBasedRasterizationGrad->set_D_vertices(						(float3*)			d_inputVertexPos						+ b * numberOfPoints);
        cudaBasedRasterizationGrad->set_D_vertexColors(					(float3*)			d_inputVertexColor						+ b * numberOfPoints);
        cudaBasedRasterizationGrad->set_D_textureMap(										d_inputTexture							+ b * textureResolutionV * textureResolutionU * 3);

        cudaBasedRasterizationGrad->set_D_shCoeff(											d_inputSHCoeff							+ b * numberOfCameras * 27);
        cudaBasedRasterizationGrad->set_D_vertexNormal(					(float3*)			d_inputVertexNormal						+ b * numberOfCameras * numberOfPoints);
        cudaBasedRasterizationGrad->set_D_barycentricCoordinatesBuffer( (float2 *)			d_inputBaryCentricBuffer				+ b * numberOfCameras * renderResolutionV * renderResolutionU);

        cudaBasedRasterizationGrad->set_D_faceIDBuffer(					(int*)				d_inputFaceBuffer						+ b * numberOfCameras * renderResolutionV * renderResolutionU);
        cudaBasedRasterizationGrad->set_D_targetImage(										d_inputTargetImage						+ b * numberOfCameras * renderResolutionV * renderResolutionU * 3);
        cudaBasedRasterizationGrad->set_D_extrinsics(										d_inputExtrinsics						+ b * numberOfCameras * 12);
        cudaBasedRasterizationGrad->set_D_intrinsics(										d_inputIntrinsics						+ b * numberOfCameras * 9);

        //set output
        cudaBasedRasterizationGrad->set_D_vertexPosGrad(				(float3*)			d_outputVertexPosGrad					+ b * numberOfPoints);
        cudaBasedRasterizationGrad->set_D_vertexColorGrad(				(float3*)			d_outputVertexColorGrad					+ b * numberOfPoints);
        cudaBasedRasterizationGrad->set_D_textureGrad(					(float3*)			d_outputTextureGrad						+ b * textureResolutionU * textureResolutionV);
        cudaBasedRasterizationGrad->set_D_shCoeffGrad(					(float*)			d_outputSHCoeffGrad						+ b * numberOfCameras * 27);

        //get gradients
        cudaBasedRasterizationGrad->renderBuffersGrad();
    }
}

//==============================================================================================//

REGISTER_KERNEL_BUILDER(Name("CudaRendererGradGpu").Device(DEVICE_GPU), CudaRendererGrad);
