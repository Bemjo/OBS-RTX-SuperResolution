#include <obs-module.h>
#include <plugin-support.h>
#include <util/threading.h>
#include <dxgi.h>
#include <d3d11.h>
#include <d3d11_1.h>
#include <tchar.h>
#include "include/nvvfx.h"



#define do_log(level, format, ...) obs_log(level, format, ##__VA_ARGS__)
#define warn(format, ...) do_log(LOG_WARNING, format, ##__VA_ARGS__)
#define info(format, ...) do_log(LOG_INFO, format, ##__VA_ARGS__)
#define error(format, ...) do_log(LOG_ERROR, format, ##__VA_ARGS__)

#ifdef _DEBUG
#define debug(format, ...) do_log(LOG_DEBUG, format, ##__VA_ARGS__)
#else
#define debug(format, ...)
#endif



/* -------------------------------------------------------- */


#define S_TYPE "type"
#define S_TYPE_NONE 0
#define S_TYPE_SR 1
#define S_TYPE_UP 2

#define S_ENABLE_AR "ar"
#define S_MODE_AR "ar_mode"
#define S_MODE_SR "sr_mode"
#define S_MODE_WEAK 0
#define S_MODE_STRONG 1

#define S_SCALE "scale"
#define S_SCALE_NONE 0
#define S_SCALE_133x 1
#define S_SCALE_15x 2
#define S_SCALE_2x 3
#define S_SCALE_3x 4
#define S_SCALE_4x 5
#define S_SCALE_N 6

#define S_STRENGTH "strength"
#define S_STRENGTH_DEFAULT 0.4f

#define S_INVALID_WARNING "warning"
#define S_INVALID_WARNING_AR "warning_ar"
#define S_INVALID_WARNING_SR "warning_sr"

#define MT_ obs_module_text
#define TEXT_FILTER MT_("SuperResolution.Filter")
#define TEXT_FILTER_NONE MT_("SuperResolution.Filter.None")
#define TEXT_FILTER_SR MT_("SuperResolution.Filter.SuperRes")
#define TEXT_FILTER_UP MT_("SuperResolution.Filter.Upscaling")
#define TEXT_SR_MODE MT_("SuperResolution.SRMode")
#define TEXT_SR_MODE_WEAK MT_("SuperResolution.SRMode.Weak")
#define TEXT_SR_MODE_STRONG MT_("SuperResolution.SRMode.Strong")
#define TEXT_AR MT_("SuperResolution.AR")
#define TEXT_AR_DESC MT_("SuperResolution.ARDesc")
#define TEXT_AR_MODE MT_("SuperResolution.ARMode")
#define TEXT_AR_MODE_WEAK MT_("SuperResolution.ARMode.Weak")
#define TEXT_AR_MODE_STRONG MT_("SuperResolution.ARMode.Strong")
#define TEXT_UP_STRENGTH MT_("SuperResolution.Strength")
#define TEXT_SCALE MT_("SuperResolution.Scale")
#define TEXT_SCALE_SIZE_133x MT_("SuperResolution.Scale.133")
#define TEXT_SCALE_SIZE_15x MT_("SuperResolution.Scale.15")
#define TEXT_SCALE_SIZE_2x MT_("SuperResolution.Scale.2")
#define TEXT_SCALE_SIZE_3x MT_("SuperResolution.Scale.3")
#define TEXT_SCALE_SIZE_4x MT_("SuperResolution.Scale.4")
#define TEXT_INVALID_WARNING MT_("SuperResolution.Invalid")
#define TEXT_INVALID_WARNING_AR MT_("SuperResolution.InvalidAR")
#define TEXT_INVALID_WARNING_SR MT_("SuperResolution.InvalidSR")

/* Set at module load time, checks to see if the NvVFX SDK is loaded, and what the users GPU and drivers supports */
/* Usable everywhere except load_nv_superresolution_filter */
static bool nvvfx_loaded = false;
static bool nvvfx_supports_ar = false;
static bool nvvfx_supports_sr = false;
static bool nvvfx_supports_up = false;
static obs_property_t *g_invalid_warning = NULL;
static obs_property_t *g_invalid_warning_ar = NULL;
static obs_property_t *g_invalid_warning_sr = NULL;

// while the filter allows for non 16:9 aspect ratios, these 16:9 values are used to validate input source sizes
// so even though a 4:3 source may be provided that has the same pixel count as a 16:9 source -
// if the resolution is outside these bounds it will be deemed invalid for processing
// see https://docs.nvidia.com/deeplearning/maxine/vfx-sdk-programming-guide/index.html#super-res-filter
static const uint32_t nv_type_resolutions[S_SCALE_N][2][2] =
{
	{{160, 90}, {1920, 1080}}, // S_SCALE_NONE
	{{160, 90}, {3840, 2160}}, // S_SCALE_133x
	{{160, 90}, {3840, 2160}}, // S_SCALE_15x
	{{160, 90}, {1920, 1080}}, // S_SCALE_2x
	{{160, 90}, {1280, 720}},  // S_SCALE_3x
	{{160, 90}, {960, 540}}    // S_SCALE_4x
};



struct nv_superresolution_data
{
	/* OBS and other vars */
	volatile bool processing_stopped; // essentially a mutex to use, as a signal to stop processing of things outside of our block
									// or to signal a catastrophic failure has occured
	obs_source_t *context;
	bool processed_frame;
	bool done_initial_render;
	bool is_target_valid;
	bool show_size_error;
	bool got_new_frame;
	bool alloc_render_tex;
	signal_handler_t *handler;
	bool reload_ar_fx;
	bool reload_sr_fx;
	bool reload_up_fx;
	uint32_t target_width;	// The current real width of our source target, may be 0
	uint32_t target_height;	// The current real height of our source target, may be 0
	bool apply_ar;
	bool are_ar_buffers_valid;
	bool are_sr_buffers_valid;
	bool are_up_buffers_valid;
	bool are_images_allocated;

	/* RTX SDK vars */
	unsigned int version;
	NvVFX_Handle sr_handle;
	NvVFX_Handle up_handle;
	NvVFX_Handle ar_handle;
	CUstream stream;	// CUDA stream
	int ar_mode;		// filter mode, should be one of S_MODE_AR
	int sr_mode;		// filter mode, should be one of S_MODE_SR
	int type;			// filter type, should be one of S_TYPE_
	int scale;			// scale mode, should be one of S_SCALE_
	float strength;		// effect strength, only effects upscaling filter?

	/* OBS render buffers for NvVFX */
	NvCVImage *src_img; // src img in obs format (RGBA) on GPU pointing to a live d3d11 gs_texture used by obs
	NvCVImage *dst_img; // the final processed image, in obs format (RGBA)  pointing to a live d3d11 gs_texture used by obs

	/* Artifact Reduction Buffers in BGRf32 Planar format */
	NvCVImage *gpu_ar_src_img;
	NvCVImage *gpu_ar_dst_img;

	/* Super Resolution buffers in either BGRf32 Planar or Upscaling buffers in RGBAu8 Chunky format */
	NvCVImage *gpu_sr_src_img; // src img in appropriate filter format on GPU
	NvCVImage *gpu_sr_dst_img; // final processed image in appropriate filter format on gpu

	NvCVImage *gpu_up_src_img; // src img in RGBAu8
	NvCVImage *gpu_up_dst_img; // final processed image in RGBAu8
	
	/* A staging buffer that is the maximal size for the selected filters to avoid allocations during transfers */
	NvCVImage *gpu_staging_img; // RGBAu8 Chunky if Upscaling only, BGRf32 otherwise

	/* Intermediate buffer between final destination image, and dst_img.
	* This shouldn't be needed, but for some reason I get a pixelformat error trying to transfer between the final dst_img
	* which should only happen when trying to transfer between incompatible formats
	* but the transfers are only BGRf32 planar <----> RGBAu8 chunky, or between exact same formats, which are all fully supported
	* See Table 4, Pixel Conversions https://docs.nvidia.com/deeplearning/maxine/nvcvimage-api-guide/index.html#nvcvimage-transfer */
	NvCVImage *gpu_dst_tmp_img; // RGBAu8 chunky Format

	/* upscaling effect vars */
	gs_effect_t *effect;
	gs_texrender_t *render;		  // TODO: remove this, and just render directly to the render_unorm
	gs_texrender_t *render_unorm; // the converted RGBA U8 render of our source
	gs_texture_t *scaled_texture; // the final RGBA U8 processed texture of the filter
	uint32_t width;         // width of source
	uint32_t height;        // height of source
	uint32_t out_width;     // output width determined by filter
	uint32_t out_height;	// output height determined by filter
	enum gs_color_space space;
	gs_eparam_t *image_param;
	gs_eparam_t *upscaled_param;	// TODO: Remove this, and just use the default OBS rendering effect
	gs_eparam_t *multiplier_param;	// TODO: remove this, and just use the default OBS rendering effect
};



typedef struct img_create_params
{
	NvCVImage **buffer;
	uint32_t width, height;
	uint32_t width2, height2;
	uint32_t layout;
	uint32_t alignment;
	NvCVImage_PixelFormat pixel_fmt;
	NvCVImage_ComponentType comp_type;
} img_create_params_t;



static void nv_sdk_path(TCHAR *buffer, size_t len)
{
	/* Currently hardcoded to find windows install directory, as that is the only supported OS supported by NvVFX */
#ifndef _WIN32
#error Platform not supported, at this time the nVidia Maxine VFX SDK only supports Windows
#endif
	TCHAR path[MAX_PATH];

	// There can be multiple apps on the system,
	// some might include the SDK in the app package and
	// others might expect the SDK to be installed in Program Files
	GetEnvironmentVariable(TEXT("NV_VIDEO_EFFECTS_PATH"), path, MAX_PATH);

	if (_tcscmp(path, TEXT("USE_APP_PATH")))
	{
		// App has not set environment variable to "USE_APP_PATH"
		// So pick up the SDK dll and dependencies from Program Files
		GetEnvironmentVariable(TEXT("ProgramFiles"), path, MAX_PATH);
		_stprintf_s(buffer, len, TEXT("%s\\NVIDIA Corporation\\NVIDIA Video Effects\\"), path);
	}
}



static void get_nvfx_sdk_path(char *buffer, size_t len)
{
	TCHAR tbuffer[MAX_PATH];
	TCHAR tmodelDir[MAX_PATH];

	nv_sdk_path(tbuffer, MAX_PATH);

	size_t max_len = sizeof(tbuffer) / sizeof(TCHAR);
	_snwprintf_s(tmodelDir, max_len, max_len, TEXT("%s\\models"), tbuffer);

	wcstombs_s(0, buffer, len, tmodelDir, MAX_PATH);
}



static inline void get_scale_factor(uint32_t s_scale, uint32_t in_x, uint32_t in_y, uint32_t *out_x, uint32_t *out_y)
{
	float scale = 1.0f;

	switch (s_scale)
	{
	case S_SCALE_133x:
		scale = 4.0f/3.0f;
		break;
	case S_SCALE_15x:
		scale = 1.5f;
		break;
	case S_SCALE_2x:
		scale = 2.0f;
		break;
	case S_SCALE_3x:
		scale = 3.0f;
		break;
	case S_SCALE_4x:
		scale = 4.0f;
		break;
	}

	*out_x = (uint32_t)(in_x * scale + 0.5f);
	*out_y = (uint32_t)(in_y * scale + 0.5f);
}



static inline bool validate_source_size(uint32_t scale, uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2)
{
	/* validate input/output aspect ratios match */
	if ((x1 * y2) != (y1 * x2))
	{
		return false;
	}

	uint32_t min_width = nv_type_resolutions[scale][0][0];
	uint32_t max_width = nv_type_resolutions[scale][1][0];
	uint32_t min_height = nv_type_resolutions[scale][0][1];
	uint32_t max_height = nv_type_resolutions[scale][1][1];

	return (x1 >= min_width && x1 <= max_width && y1 >= min_height && y1 <= max_height);
}



static void nv_superres_filter_actual_destroy(void *data)
{
	struct nv_superresolution_data *filter =
		(struct nv_superresolution_data *)data;

	if (!nvvfx_loaded) {
		bfree(filter);
		return;
	}

	os_atomic_set_bool(&filter->processing_stopped, true);

	obs_enter_graphics();

	if (filter->scaled_texture) {
		gs_texture_destroy(filter->scaled_texture);
		filter->scaled_texture = NULL;
	}
	if (filter->render) {
		gs_texrender_destroy(filter->render);
		filter->render = NULL;
	}
	if (filter->render_unorm) {
		gs_texrender_destroy(filter->render_unorm);
		filter->render_unorm = NULL;
	}

	obs_leave_graphics();

	if (filter->src_img) {
		NvCVImage_Destroy(filter->src_img);
		filter->src_img = NULL;
	}
	if (filter->gpu_sr_src_img) {
		NvCVImage_Destroy(filter->gpu_sr_src_img);
		filter->gpu_sr_src_img = NULL;
	}
	if (filter->gpu_sr_dst_img) {
		NvCVImage_Destroy(filter->gpu_sr_dst_img);
		filter->gpu_sr_dst_img = NULL;
	}
	if (filter->gpu_up_src_img) {
		NvCVImage_Destroy(filter->gpu_up_src_img);
		filter->gpu_up_src_img = NULL;
	}
	if (filter->gpu_up_dst_img) {
		NvCVImage_Destroy(filter->gpu_up_dst_img);
		filter->gpu_up_dst_img = NULL;
	}
	if (filter->gpu_dst_tmp_img) {
		NvCVImage_Destroy(filter->gpu_dst_tmp_img);
		filter->gpu_dst_tmp_img = NULL;
	}
	if (filter->dst_img) {
		NvCVImage_Destroy(filter->dst_img);
		filter->dst_img = NULL;
	}
	if (filter->gpu_staging_img) {
		NvCVImage_Destroy(filter->gpu_staging_img);
		filter->gpu_staging_img = NULL;
	}
	if (filter->gpu_ar_src_img) {
		NvCVImage_Destroy(filter->gpu_ar_src_img);
		filter->gpu_ar_src_img = NULL;
	}
	if (filter->gpu_ar_dst_img) {
		NvCVImage_Destroy(filter->gpu_ar_dst_img);
		filter->gpu_ar_dst_img = NULL;
	}

	if (filter->stream) {
		NvVFX_CudaStreamDestroy(filter->stream);
		filter->stream = NULL;
	}
	if (filter->sr_handle) {
		NvVFX_DestroyEffect(filter->sr_handle);
		filter->sr_handle = NULL;
	}
	if (filter->up_handle) {
		NvVFX_DestroyEffect(filter->up_handle);
		filter->up_handle = NULL;
	}
	if (filter->ar_handle) {
		NvVFX_DestroyEffect(filter->ar_handle);
		filter->ar_handle = NULL;
	}

	if (filter->effect) {
		obs_enter_graphics();	
		gs_effect_destroy(filter->effect);
		filter->effect = NULL;
		obs_leave_graphics();
	}

	bfree(filter);
}



static void nv_superres_filter_destroy(void *data)
{
	obs_queue_task(OBS_TASK_GRAPHICS, nv_superres_filter_actual_destroy, data, false);
}



/* Macro shenanigans to deal with variadic arguments to the error */
#define kill_error(msg, filter, ...){obs_log(LOG_ERROR, msg, ##__VA_ARGS__);os_atomic_set_bool(&filter->processing_stopped, true);}

#define nv_error(vfxErr, msg, filter, destroy_filter, ...) {	\
	if (NVCV_SUCCESS != vfxErr)\
	{\
		filter = (struct nv_superresolution_data*)filter;			\
		const char *errString = NvCV_GetErrorStringFromCode(vfxErr);\
		obs_log(LOG_ERROR, msg, ##__VA_ARGS__);						\
		obs_log(LOG_ERROR, "NvVFX Error %i: %s", vfxErr, errString);\
		if (destroy_filter) {nv_superres_filter_destroy(filter);}	\
		else {os_atomic_set_bool(&filter->processing_stopped, true);}\
		return false;\
	}\
}

/* no return variation */
#define nv_error_nr(vfxErr, msg, filter, destroy_filter, ...)\
{															\
	if (NVCV_SUCCESS != vfxErr) {                           \
		filter = (struct nv_superresolution_data *)filter;	\
		const char *errString =                             \
			NvCV_GetErrorStringFromCode(vfxErr);			\
		obs_log(LOG_ERROR, msg, ##__VA_ARGS__);             \
		obs_log(LOG_ERROR, "NvVFX Error %i: %s", vfxErr,    \
			errString);										\
		if (destroy_filter) {                               \
			nv_superres_filter_destroy(filter);				\
		} else {                                            \
			os_atomic_set_bool(								\
				&filter->processing_stopped, true);			\
		}                                                   \
	}                                                       \
}



static bool create_nvfx(struct nv_superresolution_data *filter, NvVFX_Handle *handle, NvVFX_EffectSelector fx, bool set_model_dir)
{
	if (*handle)
	{
		NvVFX_DestroyEffect(*handle);
	}

	NvCV_Status vfxErr = NvVFX_CreateEffect(fx, handle);
	nv_error(vfxErr, "Error creating nVidia RTX Upscaling FX", filter, true);

	if (set_model_dir)
	{
		char model_dir[MAX_PATH];
		get_nvfx_sdk_path(model_dir, MAX_PATH);

		vfxErr = NvVFX_SetString(*handle, NVVFX_MODEL_DIRECTORY, model_dir);
		nv_error(vfxErr, "Error seting Super Resolution model directory: [%s]", filter, true, model_dir);
	}

	vfxErr = NvVFX_SetCudaStream(*handle, NVVFX_CUDA_STREAM, filter->stream);
	nv_error(vfxErr, "Error seting Super Resolution CUDA stream", filter, true);

	return true;
}



static bool nv_load_ar_fx(struct nv_superresolution_data *filter)
{
	NvCV_Status vfxErr = NvVFX_Load(filter->ar_handle);

	if (NVCV_SUCCESS != vfxErr)
	{
		filter->are_ar_buffers_valid = false;

		if (NVCV_ERR_RESOLUTION != vfxErr)
		{
			const char *errString = NvCV_GetErrorStringFromCode(vfxErr);
			error("Failed to load NvVFX AR effect %i: %s", vfxErr, errString);
			os_atomic_set_bool(&filter->processing_stopped, true);
			return false;
		}

		obs_property_set_visible(g_invalid_warning_ar, true);
		filter->reload_ar_fx = false;
		return false;
	}

	obs_property_set_visible(g_invalid_warning_ar, false);
	filter->reload_ar_fx = false;
	filter->are_ar_buffers_valid = true;


	return true;
}



static bool nv_load_sr_fx(struct nv_superresolution_data *filter)
{
	NvCV_Status vfxErr = NVCV_SUCCESS;

	vfxErr = NvVFX_Load(filter->sr_handle);
	if (NVCV_SUCCESS != vfxErr)
	{
		filter->are_sr_buffers_valid = false;

		if (NVCV_ERR_RESOLUTION != vfxErr)
		{
			const char *errString = NvCV_GetErrorStringFromCode(vfxErr);
			error("Failed to load NvVFX SR effect %i: %s", vfxErr, errString);
			os_atomic_set_bool(&filter->processing_stopped, true);
			return false;
		}

		obs_property_set_visible(g_invalid_warning_sr, true);
		filter->reload_sr_fx = false;
		return false;
	}

	obs_property_set_visible(g_invalid_warning_sr, false);
	filter->reload_sr_fx = false;
	filter->are_sr_buffers_valid = true;

	return true;
}



static bool nv_load_up_fx(struct nv_superresolution_data *filter)
{
	NvCV_Status vfxErr = NVCV_SUCCESS;

	vfxErr = NvVFX_Load(filter->up_handle);

	if (NVCV_SUCCESS != vfxErr)
	{
		filter->are_up_buffers_valid = false;

		if (NVCV_ERR_RESOLUTION != vfxErr)
		{
			const char *errString = NvCV_GetErrorStringFromCode(vfxErr);
			error("Failed to load NvVFX Upscaling effect %i: %s", vfxErr, errString);
			os_atomic_set_bool(&filter->processing_stopped, true);
			return false;
		}

		obs_property_set_visible(g_invalid_warning_sr, true);
		filter->reload_up_fx = false;
		return false;
	}

	obs_property_set_visible(g_invalid_warning_sr, false);
	filter->reload_up_fx = false;
	filter->are_up_buffers_valid = true;

	return true;
}



/* Destroys the cuda stream, and FX handles, then flags them for recreation*/
static bool nv_create_cuda(struct nv_superresolution_data *filter)
{
	if (filter->stream)
	{
		NvVFX_CudaStreamDestroy(filter->stream);
		filter->stream = NULL;
	}

	NvCV_Status vfxErr = NvVFX_CudaStreamCreate(&filter->stream);
	nv_error(vfxErr, "Failed to create NvVFX CUDA Stream: %i", filter, true);

	return true;
}



/*
* Applies user settings changes to the filter, setting update flags.
* These changes are processed inside the render loop.
*/
static void nv_superres_filter_update(void *data, obs_data_t *settings)
{
	struct nv_superresolution_data *filter =(struct nv_superresolution_data *)data;

	filter->type = (int)obs_data_get_int(settings, S_TYPE);
	filter->apply_ar = obs_data_get_bool(settings, S_ENABLE_AR);
	filter->scale = (int)obs_data_get_int(settings, S_SCALE);

	int sr_mode = (int)obs_data_get_int(settings, S_MODE_SR);
	if (filter->sr_mode != sr_mode)
	{
		filter->sr_mode = sr_mode;
		filter->reload_sr_fx = true;

		NvCV_Status vfxErr = NvVFX_SetU32(filter->sr_handle, NVVFX_MODE, filter->sr_mode);
		nv_error_nr(vfxErr,"Failed to set SR mode", filter, false);
	}

	int ar_mode = (int)obs_data_get_int(settings, S_MODE_AR);
	if (filter->ar_mode != ar_mode)
	{
		filter->ar_mode = ar_mode;
		filter->reload_ar_fx = true;

		NvCV_Status vfxErr = NvVFX_SetU32(filter->ar_handle, NVVFX_MODE, filter->ar_mode);
		nv_error_nr(vfxErr, "Failed to set AR mode", filter, false);
	}

	float strength = (float)obs_data_get_double(settings, S_STRENGTH);
	if (fabsf(strength - filter->strength) > EPSILON)
	{

		filter->strength = strength;
		filter->reload_up_fx = true;

		NvCV_Status vfxErr = NvVFX_SetF32(filter->up_handle, NVVFX_STRENGTH, filter->strength);
		nv_error_nr(vfxErr, "Failed to set upscaling strength", filter, false);
	}
}



static bool alloc_image_from_texture(struct nv_superresolution_data *filter, img_create_params_t *params, gs_texture_t *texture)
{
	struct ID3D11Texture2D *d11texture = (struct ID3D11Texture2D *)gs_texture_get_obj(texture);

	if (!d11texture)
	{
		error("Couldn't retrieve d3d11texture2d from gs_texture");
		return false;
	}

	NvCV_Status vfxErr;

	/* Make sure that image buffer exists first, we're going to actually (re)alloc when we init from the d3d texture */
	if (*(params->buffer) == NULL)
	{
		vfxErr = NvCVImage_Create(params->width, params->height,
					  params->pixel_fmt, params->comp_type,
					  params->layout, NVCV_GPU,
					  params->alignment, params->buffer);
		nv_error(vfxErr, "Error creating source NvCVImage", filter, false);
	}

	vfxErr = NvCVImage_InitFromD3D11Texture(*(params->buffer), d11texture);
	nv_error(vfxErr, "Error allocating NvCVImage from ID3D11Texture", filter, false);

	return true;
}



static bool alloc_image_from_texrender(struct nv_superresolution_data *filter, img_create_params_t *params, gs_texrender_t *texture)
{
	return alloc_image_from_texture(filter, params, gs_texrender_get_texture(texture));
}



/* Allocates or reallocates the NvCVImage buffer provided in the param struct
* If width2 or height 2 are > 0, the image buffer will have memory allocated to fit the maximum size
* but be sized to width X height. This is used to allocate intermediary staging buffers
* returns - false on error, true otherwise
*/
static bool alloc_image(struct nv_superresolution_data* filter, img_create_params_t *params)
{
	uint32_t create_width = params->width2 > 0 ? params->width2 : params->width;
	uint32_t create_height = params->height2 > 0 ? params->height2 : params->height;
	NvCV_Status vfx_err;

	if (*(params->buffer) != NULL)
	{
		vfx_err = NvCVImage_Realloc(
			     *(params->buffer), create_width, create_height,
			     params->pixel_fmt, params->comp_type,
			     params->layout, NVCV_GPU, params->alignment);

		nv_error(vfx_err, "Failed to re-allocate image buffer", filter, false);

	}
	else
	{
		vfx_err = NvCVImage_Create(
			     create_width, create_height, params->pixel_fmt,
			     params->comp_type, params->layout, NVCV_GPU,
			     params->alignment, params->buffer);

		nv_error(vfx_err, "Failed to create image buffer", filter, false);

		vfx_err = NvCVImage_Alloc(
			     *(params->buffer), create_width, create_height,
			     params->pixel_fmt, params->comp_type,
			     params->layout, NVCV_GPU, params->alignment);

		nv_error(vfx_err, "Failed to allocate image buffer", filter, false);

		if (create_height != params->height || create_width != params->width)
		{
			vfx_err = NvCVImage_Realloc(
				     *(params->buffer), params->width,
				     params->height,
				     params->pixel_fmt, params->comp_type,
				     params->layout, NVCV_GPU,
				     params->alignment);

			nv_error(vfx_err, "Failed to resize image buffer", filter, false);
		}
	}

	return true;
}



/* Allocates the Super Resolution source images, these are allocated anytime the target is resized */
static bool alloc_ar_images(struct nv_superresolution_data* filter)
{
	img_create_params_t ar_img =
	{
		.buffer = &filter->gpu_ar_src_img,
		.width = filter->width,
		.height = filter->height,
		.pixel_fmt = NVCV_BGR,
		.comp_type = NVCV_F32,
		.layout = NVCV_PLANAR,
		.alignment = 1,
	};

	if (!alloc_image(filter, &ar_img))
	{
		error("Failed to allocate AR source buffer");
		return false;
	}

	ar_img.buffer = &filter->gpu_ar_dst_img;

	if (!alloc_image(filter, &ar_img))
	{
		error("Failed to allocate AR dest buffer");
		return false;
	}

	NvCV_Status vfxErr = NvVFX_SetImage(filter->ar_handle, NVVFX_INPUT_IMAGE, filter->gpu_ar_src_img);
	nv_error(vfxErr, "Failed to set input image for Artifact Reduction filter", filter, false);

	vfxErr = NvVFX_SetImage(filter->ar_handle, NVVFX_OUTPUT_IMAGE, filter->gpu_ar_dst_img);
	nv_error(vfxErr, "Failed to set output image for Artifact Reduction filter", filter, false);

	filter->reload_ar_fx = true;

	return true;
}



static bool alloc_obs_textures(struct nv_superresolution_data* filter)
{
	/* 3. create texrenders */
	if (filter->render)
	{
		gs_texrender_destroy(filter->render);
	}

	filter->render = gs_texrender_create(gs_get_format_from_space(filter->space), GS_ZS_NONE);

	if (!filter->render)
	{
		kill_error("Failed to create render texrenderer", filter);
		return false;
	}

	if (filter->render_unorm)
	{
		gs_texrender_destroy(filter->render_unorm);
	}

	filter->render_unorm = gs_texrender_create(GS_BGRA_UNORM, GS_ZS_NONE);

	if (!filter->render_unorm)
	{
		kill_error("Failed to create render_unorm texrenderer", filter);
		return false;
	}

	filter->done_initial_render = false;
	filter->alloc_render_tex = false;

	return true;
}



/* Allocates the Super Resolution source images, these are allocated anytime the target is resized, or the filter type changed */
static bool alloc_sr_source_images(struct nv_superresolution_data *filter)
{
	if (!filter->is_target_valid)
	{
		return true;
	}

	img_create_params_t img = {
		.buffer = &filter->gpu_sr_src_img,
		.width = filter->width,
		.height = filter->height,
		.pixel_fmt = NVCV_BGR,
		.comp_type = NVCV_F32,
		.layout = NVCV_PLANAR,
		.alignment = 1,
	};

	if (!alloc_image(filter, &img))
	{
		error("Failed to allocate SuperRes source buffer");
		return false;
	}

	img.buffer = &filter->gpu_up_src_img,
	img.pixel_fmt = NVCV_RGBA;
	img.comp_type = NVCV_U8;
	img.layout = NVCV_CHUNKY;
	img.alignment = 32;

	if (!alloc_image(filter, &img))
	{
		error("Failed to allocate Upscaling source buffer");
		return false;
	}

	NvCV_Status vfxErr = NvVFX_SetImage(filter->sr_handle, NVVFX_INPUT_IMAGE, filter->gpu_sr_src_img);
	nv_error(vfxErr, "Error setting SuperRes input image", filter, false);

	vfxErr = NvVFX_SetImage(filter->up_handle, NVVFX_INPUT_IMAGE, filter->gpu_up_src_img);
	nv_error(vfxErr, "Error setting Upscaling input image", filter, false);

	filter->reload_sr_fx = true;
	filter->reload_up_fx = true;

	return true;
}



/* Allocates the Super Resolution source images, these are allocated anytime the target is resized, the filter type changed, or the scale changed */
static bool alloc_sr_dest_images(struct nv_superresolution_data* filter)
{
	if (!filter->is_target_valid)
	{
		return true;
	}

	/* Allocate the destination first */
	img_create_params_t sr_img = {
		.buffer = &filter->gpu_sr_dst_img,
		.width = filter->out_width,
		.height = filter->out_height,
		.pixel_fmt = NVCV_BGR,
		.comp_type = NVCV_F32,
		.layout = NVCV_PLANAR,
		.alignment = 1,
	};

	if (!alloc_image(filter, &sr_img))
	{
		error("Failed to allocate NvCVImage SR dest buffer");
		return false;
	}

	/* Allocate the staging buffer next to set it's size */
	sr_img.buffer = &filter->gpu_staging_img;
	sr_img.width = filter->width;
	sr_img.height = filter->height;
	sr_img.width2 = filter->out_width;
	sr_img.height2 = filter->out_height;

	if (!alloc_image(filter, &sr_img))
	{
		error("Failed to allocate NvCVImage FX staging buffer");
		return false;
	}

	/* Finally allocate space for the final result temporary transfer buffer */
	sr_img.buffer = &filter->gpu_dst_tmp_img;
	sr_img.pixel_fmt = NVCV_RGBA;
	sr_img.comp_type = NVCV_U8;
	sr_img.layout = NVCV_CHUNKY;
	sr_img.alignment = 0;
	sr_img.width = filter->out_width;
	sr_img.height = filter->out_height;
	sr_img.width2 = 0;
	sr_img.height2 = 0;

	if (!alloc_image(filter, &sr_img))
	{
		error("Failed to allocate upscaled NvCVImage bufer");
		return false;
	}

	/* Allocate space for the upscaling destination */
	sr_img.buffer = &filter->gpu_up_dst_img;
	sr_img.pixel_fmt = NVCV_RGBA;
	sr_img.comp_type = NVCV_U8;
	sr_img.layout = NVCV_CHUNKY;
	sr_img.alignment = 32;
	sr_img.width = filter->out_width;
	sr_img.height = filter->out_height;
	sr_img.width2 = 0;
	sr_img.height2 = 0;

	if (!alloc_image(filter, &sr_img))
	{
		error("Failed to allocate upscaling destination buffer");
		return false;
	}

	NvCV_Status vfxErr = NvVFX_SetImage(filter->sr_handle, NVVFX_OUTPUT_IMAGE, filter->gpu_sr_dst_img);
	nv_error(vfxErr, "Error setting SuperRes output image", filter, false);

	vfxErr = NvVFX_SetImage(filter->up_handle, NVVFX_OUTPUT_IMAGE, filter->gpu_up_dst_img);
	nv_error(vfxErr, "Error setting Upscaling output image", filter, false);

	filter->reload_sr_fx = true;
	filter->reload_up_fx = true;

	return true;
}



/* (Re)allocates any images that are pending (re)allocation
/* @return - false if there's any error, true otherwise */
static bool alloc_nvfx_images(struct nv_superresolution_data *filter)
{
	if (!alloc_ar_images(filter))
	{
		error("Failed to allocate AR NvFXImages");
		return false;
	}

	if (!alloc_sr_source_images(filter))
	{
		error("Failed to allocate SR Source NvFXImages");
		return false;
	}

	if (!alloc_sr_dest_images(filter))
	{
		error("Failed to allocate SR Dest NvFXImages");
		return false;
	}

	return true;
}



/* Destroys the cuda stream, and all FX, and reloads them */
static bool init_cuda_and_fx(struct nv_superresolution_data *filter)
{
	filter->reload_ar_fx = true;
	filter->reload_sr_fx = true;
	filter->reload_up_fx = true;

	return nv_create_cuda(filter) &&
			create_nvfx(filter, &filter->ar_handle, NVVFX_FX_ARTIFACT_REDUCTION, true) &&
			create_nvfx(filter, &filter->sr_handle, NVVFX_FX_SUPER_RES, true) &&
			create_nvfx(filter, &filter->up_handle, NVVFX_FX_SR_UPSCALE, false);
}



static bool alloc_destination_image(struct nv_superresolution_data* filter)
{
	if (filter->scaled_texture)
	{
		gs_texture_destroy(filter->scaled_texture);
	}

	filter->scaled_texture = gs_texture_create(filter->out_width, filter->out_height, GS_RGBA_UNORM, 1, NULL, 0);

	if (!filter->scaled_texture)
	{
		kill_error("Final output texture couldn't be created", filter);
		return false;
	}

	img_create_params_t params = {
		.buffer = &filter->dst_img,
		.width = filter->out_width,
		.height = filter->out_height,
		.pixel_fmt = NVCV_RGBA,
		.comp_type = NVCV_U8,
		.layout = NVCV_CHUNKY,
		.alignment = 0
	};

	if (!alloc_image_from_texture(filter, &params, filter->scaled_texture))
	{
		error("Failed to create dest NvCVImage from OBS output texture");
		return false;
	}

	return true;
}



/* Allocates any textures or images that have been flagged for allocation
* Used in both initialization and render tick to ensure things are created before use */
static bool init_images(struct nv_superresolution_data* filter)
{
	if (!alloc_obs_textures(filter))
	{
		return false;
	}

	if (!alloc_nvfx_images(filter))
	{
		return false;
	}

	if (!alloc_destination_image(filter))
	{
		return false;
	}

	filter->are_images_allocated = true;

	return true;
}



static void nv_superres_filter_reset(void *data, calldata_t *calldata)
{
	struct nv_superresolution_data *filter = (struct nv_superresolution_data *)data;

	os_atomic_set_bool(&filter->processing_stopped, true);

	if (!init_cuda_and_fx(filter))
	{
		return;
	}

	filter->are_images_allocated = false;

	os_atomic_set_bool(&filter->processing_stopped, false);
}



static bool process_texture_superres(struct nv_superresolution_data *filter)
{
	float scale = 1.0f/255.0f;
	float inv_scale = 255.0f;

	bool do_ar_pass = filter->apply_ar && filter->are_ar_buffers_valid;
	bool do_sr_pass =
		(filter->type == S_TYPE_UP && filter->are_up_buffers_valid) ||
		(filter->type == S_TYPE_SR && filter->are_sr_buffers_valid);

	/* From nVidias recommendations here https://docs.nvidia.com/deeplearning/maxine/vfx-sdk-programming-guide/index.html#upscale-filter
	* We have 3 main paths to take
	* A. AR pass only
	* B. Upscaling Pass Only
	* C. AR Pass -> Upscaling Pass
	* So the effect pipeline is
	*	A: src_img -> staging -> AR_src -> Run FX -> AR_dst -> staging											-> dst_tmp_img -> staging -> dst_img
	*	B: src_img -> staging -> SR_src -> Run FX -> SR_dst -> staging											-> dst_tmp_img -> staging -> dst_img
	*	C: src_img -> staging -> AR_src -> Run FX -> AR_dst -> staging -> SR_src -> Run FX -> SR_dst -> staging	-> dst_tmp_img -> staging -> dst_img
	*/

	NvCVImage *destination = filter->gpu_dst_tmp_img;
	
	if (do_ar_pass)
	{
		destination = filter->gpu_ar_src_img;
	}
	else if (do_sr_pass)
	{
		switch (filter->type)
		{
		case S_TYPE_UP:
			destination = filter->gpu_up_src_img;
			break;
		case S_TYPE_SR:
			destination = filter->gpu_sr_src_img;
			break;
		}
	}

	if (!do_ar_pass && do_sr_pass && filter->type == S_TYPE_UP)
	{
		scale = 1.0f;
		inv_scale = 1.0f;
	}

	/* Do transfer of src_img to first stage source */
	NvCV_Status vfxErr = NvCVImage_MapResource(filter->src_img, filter->stream);
	nv_error(vfxErr, "Error mapping resource for source texture", filter, false);

	vfxErr = NvCVImage_Transfer(filter->src_img, destination, scale, filter->stream, filter->gpu_staging_img);
	nv_error(vfxErr, "Error converting src img for first filter pass", filter, false);

	vfxErr = NvCVImage_UnmapResource(filter->src_img, filter->stream);
	nv_error(vfxErr, "Error unmapping resource for src texture", filter, false);

	/* 2. process artifact reduction fx pass, and transfer to upscaling pass, or to final dst_tmp_img */
	if (do_ar_pass)
	{
		vfxErr = NvVFX_Run(filter->ar_handle, 1);

		if (vfxErr == NVCV_ERR_CUDA)
		{
			nv_superres_filter_reset(filter, NULL);
			return false;
		}

		nv_error(vfxErr, "Error running the AR FX", filter, false);

		// We're transfering to the upscaling stage, or we're passing through to the final result
		switch (filter->type)
		{
			case S_TYPE_NONE:
				destination = filter->gpu_dst_tmp_img;
				break;
			case S_TYPE_UP:
				destination = filter->gpu_up_src_img;
				break;
			case S_TYPE_SR:
				destination = filter->gpu_sr_src_img;
				break;
		}

		float iscale = (!do_sr_pass || (do_sr_pass && filter->type == S_TYPE_UP)) ? 255.0f : 1.0f;

		vfxErr = NvCVImage_Transfer(filter->gpu_ar_dst_img, destination, iscale, filter->stream, filter->gpu_staging_img);
		nv_error(vfxErr, "Error converting src to BGR img for SR pass", filter, false);
	}

	/* 3. Run the image through the upscaling pass */
	if (do_sr_pass)
	{
		NvVFX_Handle *handle = filter->type == S_TYPE_UP ? &filter->up_handle : &filter->sr_handle;

		vfxErr = NvVFX_Run(*handle, 1);

		if (vfxErr == NVCV_ERR_CUDA)
		{
			nv_superres_filter_reset(filter, NULL);
			return false;
		}

		nv_error(vfxErr, "Error running the NvVFX Super Resolution stage.", filter, false);

		NvCVImage *source = filter->type == S_TYPE_UP ? filter->gpu_up_dst_img : filter->gpu_sr_dst_img;
		float iscale = filter->type == S_TYPE_UP ? 1.0f : 255.0f;

		/* 3.5 move to a temp buffer, not tied to a bound D3D11 gs_texture_t, or used as an input/output NvCVImage to an effect */
		vfxErr = NvCVImage_Transfer(source, filter->gpu_dst_tmp_img, iscale, filter->stream, filter->gpu_staging_img);
		nv_error(vfxErr, "Error transfering super resolution upscaled texture to temporary buffer", filter, false);
	}

	/* 4. Do the final dst_tmp_img -> staging -> dst_img transfer */
	vfxErr = NvCVImage_MapResource(filter->dst_img, filter->stream);
	nv_error(vfxErr, "Error mapping resource for dst texture", filter, false);

	vfxErr = NvCVImage_Transfer(filter->gpu_dst_tmp_img, filter->dst_img, 1.0f, filter->stream, filter->gpu_staging_img);
	nv_error(vfxErr, "Error transferring temporary image buffer to final dest buffer", filter, false);

	vfxErr = NvCVImage_UnmapResource(filter->dst_img, filter->stream);
	nv_error(vfxErr, "Error unmapping resource for dst texture", filter, false);

	return true;
}



/* Checks the various flags inside of filter to see if anything needs to be created, allocated, or reloaded 
* param filter - the filter structure to validate
* param inside_graphics - whether we are inside the graphics context, or whether we have to enter it
*/
static bool reload_fx(struct nv_superresolution_data* filter)
{
	if (nvvfx_supports_ar && filter->reload_ar_fx && !nv_load_ar_fx(filter))
	{
		error("Failed to load the artifact reduction NvVFX");
		return false;
	}

	if (nvvfx_supports_sr && filter->reload_sr_fx && !nv_load_sr_fx(filter))
	{
		error("Failed to load the super resolution NvVFX");
		return false;
	}

	if (nvvfx_supports_up && filter->reload_up_fx && !nv_load_up_fx(filter))
	{
		error("Failed to load the upscaling NvVFX");
		return false;
	}

	return true;
}



static void* nv_superres_filter_create(obs_data_t* settings, obs_source_t* context)
{
	struct nv_superresolution_data* filter = (struct nv_superresolution_data*)bzalloc(sizeof(*filter));

	/* Does this filter exist already on a source, but the vfx sdk libraries weren't found? Let's leave. */
	if (!nvvfx_loaded)
	{
		nv_superres_filter_destroy(filter);
		return NULL;
	}

	//NvCV_Status vfxErr;
	filter->context = context;
	filter->sr_mode = -1; // should be 0 or 1; -1 triggers an update
	filter->type = S_TYPE_SR;
	filter->show_size_error = true;
	filter->scale = S_SCALE_15x;
	filter->strength = S_STRENGTH_DEFAULT;
	os_atomic_set_bool(&filter->processing_stopped, false);

	/* Load the effect file */
	char* effect_path = obs_module_file("rtx_superresolution.effect");
	char* load_err = NULL;

	obs_enter_graphics();
	filter->effect = gs_effect_create_from_file(effect_path, &load_err);
	bfree(effect_path);

	if (filter->effect)
	{
		filter->image_param = gs_effect_get_param_by_name(filter->effect, "image");
		filter->upscaled_param = gs_effect_get_param_by_name(filter->effect, "mask");
		filter->multiplier_param = gs_effect_get_param_by_name(filter->effect, "multiplier");
	}

	obs_leave_graphics();

	if (!filter->effect)
	{
		error("Failed to load effect file: %s", load_err);
		nv_superres_filter_destroy(filter);

		if (load_err)
		{
			bfree(load_err);
		}

		return NULL;
	}

	if (!init_cuda_and_fx(filter))
	{
		error("Failed to initialize filter, couldn't create FX");
		return NULL;
	}

	nv_superres_filter_update(filter, settings);

	return filter;
}



static bool nv_filter_type_modified(obs_properties_t *ppts, obs_property_t *p, obs_data_t *settings)
{
	int type = (int)obs_data_get_int(settings, S_TYPE);

	obs_property_t *p_str = obs_properties_get(ppts, S_STRENGTH);
	obs_property_t *p_mode = obs_properties_get(ppts, S_MODE_SR);
	obs_property_t *p_scale = obs_properties_get(ppts, S_SCALE);

	if (type == S_TYPE_NONE)
	{
		obs_property_set_visible(p_str, false);
		obs_property_set_visible(p_mode, false);
		obs_property_set_visible(p_scale, false);
		return true;
	}
	else
	{
		obs_property_set_visible(p_scale, true);
	}

	bool is_upcaling = type == S_TYPE_UP;
	obs_property_set_visible(p_str, is_upcaling);
	obs_property_set_visible(p_mode, !is_upcaling);

	return true;
}



static bool ar_pass_toggled(obs_properties_t *ppts, obs_property_t *p,obs_data_t *settings)
{
	p = obs_properties_get(ppts, S_MODE_AR);
	obs_property_set_visible(p, obs_data_get_bool(settings, S_ENABLE_AR));

	return true;
}



static obs_properties_t *nv_superres_filter_properties(void *data)
{
	struct nv_superresolution_data *filter = (struct nv_superresolution_data *)data;

	obs_properties_t *props = obs_properties_create();

	obs_property_t *filter_type = obs_properties_add_list(props, S_TYPE, TEXT_FILTER, OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);

	obs_property_list_add_int(filter_type, TEXT_FILTER_NONE, S_TYPE_NONE);
	if (nvvfx_supports_sr)
		obs_property_list_add_int(filter_type, TEXT_FILTER_SR, S_TYPE_SR);
	if (nvvfx_supports_up)
		obs_property_list_add_int(filter_type, TEXT_FILTER_UP, S_TYPE_UP);

	obs_property_set_modified_callback(filter_type, nv_filter_type_modified);

	obs_property_t *scale = obs_properties_add_list(props, S_SCALE,TEXT_SCALE, OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);

	//obs_property_list_add_int(scale, TEXT_SCALE_SIZE_133x, S_SCALE_133x);	// This is so very buggy, and rarely gives supported resolutions
	obs_property_list_add_int(scale, TEXT_SCALE_SIZE_15x, S_SCALE_15x);
	obs_property_list_add_int(scale, TEXT_SCALE_SIZE_2x, S_SCALE_2x);
	obs_property_list_add_int(scale, TEXT_SCALE_SIZE_3x, S_SCALE_3x);
	obs_property_list_add_int(scale, TEXT_SCALE_SIZE_4x, S_SCALE_4x);

	if (nvvfx_supports_sr)
	{
		obs_property_t *sr_mode = obs_properties_add_list(props, S_MODE_SR, TEXT_SR_MODE, OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
		obs_property_list_add_int(sr_mode, TEXT_SR_MODE_WEAK, S_MODE_WEAK);
		obs_property_list_add_int(sr_mode, TEXT_SR_MODE_STRONG, S_MODE_STRONG);
	}

	if (nvvfx_supports_up)
	{
		obs_property_t *strength = obs_properties_add_float_slider(props, S_STRENGTH, TEXT_UP_STRENGTH, 0.0, 1.0, 0.05);
	}

	if (nvvfx_supports_ar)
	{
		obs_property_t *ar_pass = obs_properties_add_bool(props, S_ENABLE_AR, TEXT_AR_DESC);
		obs_property_set_modified_callback(ar_pass, ar_pass_toggled);

		obs_property_t *ar_modes = obs_properties_add_list(props, S_MODE_AR, TEXT_AR_MODE, OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
		obs_property_list_add_int(ar_modes, TEXT_AR_MODE_WEAK, S_MODE_WEAK);
		obs_property_list_add_int(ar_modes, TEXT_AR_MODE_STRONG, S_MODE_STRONG);
	}

	g_invalid_warning = obs_properties_add_text(props, S_INVALID_WARNING, TEXT_INVALID_WARNING, OBS_TEXT_INFO);
	obs_property_set_visible(g_invalid_warning, !filter->is_target_valid);

	g_invalid_warning_ar = obs_properties_add_text(props, S_INVALID_WARNING_AR, TEXT_INVALID_WARNING_AR, OBS_TEXT_INFO);
	obs_property_set_visible(g_invalid_warning_ar, !filter->is_target_valid);

	g_invalid_warning_sr = obs_properties_add_text(props, S_INVALID_WARNING_SR, TEXT_INVALID_WARNING_SR, OBS_TEXT_INFO);
	obs_property_set_visible(g_invalid_warning_sr, !filter->is_target_valid);

	return props;
}



static void nv_superres_filter_defaults(obs_data_t *settings)
{
	int type = nvvfx_supports_sr ? S_TYPE_SR : (nvvfx_supports_up ? S_TYPE_UP : S_TYPE_NONE);

	obs_data_set_default_int(settings, S_TYPE, type);
	obs_data_set_default_int(settings, S_SCALE, S_SCALE_15x);

	if (nvvfx_supports_ar)
	{
		obs_data_set_default_bool(settings, S_ENABLE_AR, false);
		obs_data_set_default_int(settings, S_MODE_AR, S_MODE_WEAK);
	}

	if (nvvfx_supports_sr)
	{
		obs_data_set_default_int(settings, S_MODE_SR, S_MODE_WEAK);
	}

	if (nvvfx_supports_up)
	{
		obs_data_set_default_double(settings, S_STRENGTH, S_STRENGTH_DEFAULT);
	}
}



static struct obs_source_frame *nv_superres_filter_video(void *data, struct obs_source_frame *frame)
{
	struct nv_superresolution_data *filter = (struct nv_superresolution_data *)data;
	filter->got_new_frame = true;
	return frame;
}



/*
* We check and validate our source size, requested scale size, and color space here incase they change,
* if it does we need to recreate or resize the various image buffers used to accomodate
*/
static void nv_superres_filter_tick(void *data, float t)
{
	UNUSED_PARAMETER(t);

	struct nv_superresolution_data *filter = (struct nv_superresolution_data *)data;

	if (filter->processing_stopped)
	{
		return;
	}

	obs_source_t *target = obs_filter_get_target(filter->context);

	if (!target)
	{
		return;
	}

	const uint32_t cx = obs_source_get_base_width(target);
	const uint32_t cy = obs_source_get_base_height(target);
	filter->target_width = cx;
	filter->target_height = cy;

	// initially the sizes are 0
	filter->is_target_valid = cx > 0 && cy > 0;

	if (!filter->is_target_valid)
	{
		return;
	}

	uint32_t cx_out;
	uint32_t cy_out;
	int _scale = (filter->type == S_TYPE_NONE) ? S_SCALE_NONE : filter->scale;

	get_scale_factor(_scale, cx, cy, &cx_out, &cy_out);
	filter->is_target_valid = validate_source_size(_scale, cx, cy, cx_out, cy_out);

	if (!filter->is_target_valid)
	{
		if (filter->show_size_error)
		{
			error("Input source is too small or too large for the requested scaling. Please try adding a Scale/Aspect ratio filter before this, or changing the input resolution of the source this filter is attached to!");
			filter->show_size_error = false;
			obs_property_set_visible(g_invalid_warning, true);
		}
		return;
	}
	else if (!filter->show_size_error)
	{
		obs_property_set_visible(g_invalid_warning, false);
		filter->show_size_error = true;
	}

	/* As the source size has changed, we need to flag ALL the image buffers to be reloaded */
	if (cx != filter->width || cy != filter->height || cx_out != filter->out_width || cy_out != filter->out_height)
	{
		filter->width = cx;
		filter->height = cy;
		filter->out_width = cx_out;
		filter->out_height = cy_out;
		filter->are_images_allocated = false;
	}

	if (!filter->are_images_allocated)
	{
		obs_enter_graphics();
		init_images(filter);
		obs_leave_graphics();
	}

	filter->processed_frame = false;
}



static const char *get_tech_name_and_multiplier(enum gs_color_space current_space, enum gs_color_space source_space, float *multiplier)
{
	const char *tech_name = "Draw";
	*multiplier = 1.0f;

	switch (source_space)
	{
	case GS_CS_SRGB:
	case GS_CS_SRGB_16F:
		if (current_space == GS_CS_709_SCRGB)
		{
			tech_name = "DrawMultiply";
			*multiplier = obs_get_video_sdr_white_level() / 80.0f;
		}
		break;

	case GS_CS_709_EXTENDED:
		switch (current_space)
		{
		case GS_CS_SRGB:
		case GS_CS_SRGB_16F:
			tech_name = "DrawTonemap";
			break;
		case GS_CS_709_SCRGB:
			tech_name = "DrawMultiply";
			*multiplier = obs_get_video_sdr_white_level() / 80.0f;
		}
		break;

	case GS_CS_709_SCRGB:
		switch (current_space)
		{
		case GS_CS_SRGB:
		case GS_CS_SRGB_16F:
			tech_name = "DrawMultiplyTonemap";
			*multiplier = 80.0f / obs_get_video_sdr_white_level();
			break;
		case GS_CS_709_EXTENDED:
			tech_name = "DrawMultiply";
			*multiplier = 80.0f / obs_get_video_sdr_white_level();
		}
	}

	return tech_name;
}



static void draw_superresolution(struct nv_superresolution_data *filter)
{
	/* Render alpha mask */
	const enum gs_color_space source_space = filter->space;
	float multiplier;
	const char *technique = get_tech_name_and_multiplier(gs_get_color_space(), source_space, &multiplier);
	const enum gs_color_format format = gs_get_format_from_space(source_space);

	if (obs_source_process_filter_begin_with_color_space(filter->context, format, source_space, OBS_ALLOW_DIRECT_RENDERING))
	{
		if (source_space != GS_CS_SRGB)
		{
			gs_effect_set_texture(filter->upscaled_param, filter->scaled_texture);
		}
		else
		{
			gs_effect_set_texture_srgb(filter->upscaled_param, filter->scaled_texture);
		}

		gs_effect_set_float(filter->multiplier_param, multiplier);

		gs_blend_state_push();
		gs_blend_function(GS_BLEND_ONE, GS_BLEND_INVSRCALPHA);

		obs_source_process_filter_tech_end(filter->context, filter->effect, filter->out_width, filter->out_height, technique);

		gs_blend_state_pop();
	}
}



static void render_source_to_render_tex(struct nv_superresolution_data *filter, obs_source_t *const target, obs_source_t *const parent)
{
	const uint32_t target_flags = obs_source_get_output_flags(target);
	const uint32_t parent_flags = obs_source_get_output_flags(parent);

	bool custom_draw = (target_flags & OBS_SOURCE_CUSTOM_DRAW) != 0;
	bool async = (target_flags & OBS_SOURCE_ASYNC) != 0;

	const enum gs_color_space preferred_spaces[] =
	{
		GS_CS_SRGB,
		GS_CS_SRGB_16F,
		GS_CS_709_EXTENDED,
	};

	const enum gs_color_space source_space = obs_source_get_color_space(target, OBS_COUNTOF(preferred_spaces), preferred_spaces);

	gs_texrender_t *const render = filter->render;
	gs_texrender_reset(render);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);

	if (gs_texrender_begin_with_color_space(render, filter->width, filter->height, source_space))
	{
		struct vec4 clear_color;
		vec4_zero(&clear_color);
		gs_clear(GS_CLEAR_COLOR, &clear_color, 0.0f, 0);

		gs_ortho(0.0f, (float)filter->width, 0.0f, (float)filter->height, -100.0f, 100.0f);

		if (target == parent && !custom_draw && !async)
		{
			obs_source_default_render(target);
		}
		else
		{
			obs_source_video_render(target);
		}

		gs_texrender_end(render);

		gs_texrender_t *const render_unorm = filter->render_unorm;
		gs_texrender_reset(render_unorm);

		if (gs_texrender_begin_with_color_space(render_unorm, filter->width, filter->height, GS_CS_SRGB))
		{
			const bool previous = gs_framebuffer_srgb_enabled();
			gs_enable_framebuffer_srgb(true);
			gs_enable_blending(false);

			gs_ortho(0.0f, (float)filter->width, 0.0f, (float)filter->height, -100.0f, 100.0f);

			const char *tech_name = "ConvertUnorm";
			float multiplier = 1.f;

			if (source_space == GS_CS_709_EXTENDED)
			{
				tech_name = "ConvertUnormTonemap";
			}
			else if (source_space == GS_CS_709_SCRGB)
			{
				tech_name = "ConvertUnormMultiplyTonemap";
				multiplier = 80.0f / obs_get_video_sdr_white_level();
			}

			gs_effect_set_texture_srgb(filter->image_param, gs_texrender_get_texture(render));
			gs_effect_set_float(filter->multiplier_param, multiplier);

			while (gs_effect_loop(filter->effect, tech_name))
			{
				gs_draw(GS_TRIS, 0, 3);
			}

			gs_texrender_end(render_unorm);

			gs_enable_blending(true);
			gs_enable_framebuffer_srgb(previous);
		}
	}

	gs_blend_state_pop();

	/* 2. Initialize src_texture (only at startup or reset)
	* Has to be done after we have actually rendered to a texture and it's usable */
	if (!filter->done_initial_render)
	{
		/* TODO: Maybe change this to match the input buffer type, to avoid any conversion cost
		* from the initial transfer, would have to ensure color scaling in the trasnfer is correct */
		img_create_params_t params = {
			.buffer = &filter->src_img,
			.width = filter->width,
			.height = filter->height,
			.pixel_fmt = NVCV_RGBA,
			.comp_type = NVCV_U8,
			.layout = NVCV_CHUNKY,
			.alignment = 1
		};

		filter->done_initial_render = alloc_image_from_texrender(filter, &params, filter->render_unorm);
	}
}



static void nv_superres_filter_render(void *data, gs_effect_t *effect)
{
	struct nv_superresolution_data *filter = (struct nv_superresolution_data *)data;

	if (filter->processing_stopped)
	{
		obs_source_skip_video_filter(filter->context);
		return;
	}

	obs_source_t *const target = obs_filter_get_target(filter->context);
	obs_source_t *const parent = obs_filter_get_parent(filter->context);

	/* Skip if processing of a frame hasn't yet started */
	if (!filter->is_target_valid || !target || !parent)
	{
		obs_source_skip_video_filter(filter->context);
		return;
	}

	/* We've already processed the last frame we got and we haven't seen a new one, just draw what we've already done */
	if (filter->processed_frame)
	{
		draw_superresolution(filter);
		return;
	}

	/* Ensure we've got our signal handlers setup if our source is valid */
	if (parent && !filter->handler)
	{
		filter->handler = obs_source_get_signal_handler(parent);
		signal_handler_connect(filter->handler, "update", nv_superres_filter_reset, filter);
	}

	/* Skip drawing if the user has turning everything off */
	if (!filter->apply_ar && filter->type == S_TYPE_NONE)
	{
		obs_source_skip_video_filter(filter->context);
		return;
	}
	
	/* We're waiting for the source to report a valid size for the render textures to be ready. We cannot continue until they are. */
	if (!filter->render)
	{
		obs_source_skip_video_filter(filter->context);
		return;
	}

	const enum gs_color_space preferred_spaces[] =
	{
		GS_CS_SRGB,
		GS_CS_SRGB_16F,
		GS_CS_709_EXTENDED,
	};

	const enum gs_color_space source_space = obs_source_get_color_space(target, OBS_COUNTOF(preferred_spaces), preferred_spaces);

	if (filter->space != source_space)
	{
		filter->space = source_space;
		init_images(filter);
	}

	if (!reload_fx(filter))
	{
		obs_source_skip_video_filter(filter->context);
		return;
	}

	const uint32_t target_flags = obs_source_get_output_flags(target);
	bool async = (target_flags & OBS_SOURCE_ASYNC) != 0;

	/* Render our source out to the render texture, getting it ready for the pipeline */
	render_source_to_render_tex(filter, target, parent);

	/* If we actually have a valid texture to render, process it and draw it */
	if (filter->done_initial_render && filter->are_images_allocated)
	{
		bool draw = true;

		/* should limit processing of this image */
		if (!async || filter->got_new_frame)
		{
			filter->got_new_frame = false;
			draw = process_texture_superres(filter);
		}

		if (draw)
		{
			filter->processed_frame = true;
			draw_superresolution(filter);
		}
	}
	else
	{
		obs_source_skip_video_filter(filter->context);
	}

	// TODO: Consider just using this to draw the final output instead of our custom superresolution effect
	UNUSED_PARAMETER(effect);
}



static enum gs_color_space nv_superres_filter_get_color_space(void *data, size_t count, const enum gs_color_space *preferred_spaces)
{
	const enum gs_color_space potential_spaces[] =
	{
		GS_CS_SRGB,
		GS_CS_SRGB_16F,
		GS_CS_709_EXTENDED,
	};

	struct nv_superresolution_data *const filter = (struct nv_superresolution_data *)data;

	const enum gs_color_space source_space = obs_source_get_color_space(obs_filter_get_target(filter->context), OBS_COUNTOF(potential_spaces), potential_spaces);

	enum gs_color_space space = source_space;

	for (size_t i = 0; i < count; ++i)
	{
		space = preferred_spaces[i];

		if (space == source_space)
		{
			break;
		}
	}

	return space;
}



static uint32_t nv_superres_filter_width(void *data)
{
	struct nv_superresolution_data *const filter = (struct nv_superresolution_data *)data;

	return (filter->is_target_valid && !filter->processing_stopped) ? filter->out_width : filter->target_width;
}



static uint32_t nv_superrer_filter_height(void *data)
{
	struct nv_superresolution_data *const filter = (struct nv_superresolution_data *)data;

	return (filter->is_target_valid && !filter->processing_stopped) ? filter->out_height : filter->target_height;
}



static const char *nv_superres_filter_name(void *unused)
{
	UNUSED_PARAMETER(unused);
	return obs_module_text("NVIDIASuperResolutionFilter");
}



struct obs_source_info nvidia_superresolution_filter_info =
{
	.id = "nv_superresolution_filter",
	.type = OBS_SOURCE_TYPE_FILTER,
	.output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_SRGB,
	.get_name = nv_superres_filter_name,
	.create = nv_superres_filter_create,
	.destroy = nv_superres_filter_destroy,
	.get_defaults = nv_superres_filter_defaults,
	.get_properties = nv_superres_filter_properties,
	.update = nv_superres_filter_update,
	.filter_video = nv_superres_filter_video,
	.video_render = nv_superres_filter_render,
	.video_tick = nv_superres_filter_tick,
	.video_get_color_space = nv_superres_filter_get_color_space,
	.get_width = nv_superres_filter_width,
	.get_height = nv_superrer_filter_height,
};



bool load_nv_superresolution_filter(void)
{
	const char *cstr = NULL;
	NvCV_Status err = NvVFX_GetString(NULL, NVVFX_INFO, &cstr);

	nvvfx_loaded = err == NVCV_SUCCESS;

	if (nvvfx_loaded)
	{
		if (cstr != NULL && strnlen_s(cstr, 3) > 1)
		{
			nvvfx_supports_ar = strstr(cstr, NVVFX_FX_ARTIFACT_REDUCTION) != NULL;
			nvvfx_supports_sr = strstr(cstr, NVVFX_FX_SUPER_RES) != NULL;
			nvvfx_supports_up = strstr(cstr, NVVFX_FX_SR_UPSCALE) != NULL;
		}
		obs_register_source(&nvidia_superresolution_filter_info);
	}
	else
	{
		/* We cannot load the SDK DLLs */
		if (err == NVCV_ERR_LIBRARY)
		{
			info("[NVIDIA VIDEO FX SUPERRES]: Could not load NVVFX Library, please download the video effects SDK for your GPU https://www.nvidia.com/en-us/geforce/broadcasting/broadcast-sdk/resources/");
		}
		else if (err == NVCV_ERR_UNSUPPORTEDGPU)
		{
			info("[NVIDIA VIDEO FX SUPERRES]: Unsupported GPU");
		}
		else
		{
			info("[NVIDIA VIDEO FX SUPERRES]: Error %i", err);
		}
	}
	
	return nvvfx_loaded;
}
