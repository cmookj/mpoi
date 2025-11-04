#include "mpoi.h"

#include <memory>

mpoi::mpoi ()
    : _next_key (0)
    , _src ("") {
    _setup_opencl();
}

mpoi::mpoi (const std::string& src)
    : _next_key (0)
    , _src (src) {
    _setup_opencl();
    build_program (_src);
}

mpoi::mpoi (const mpoi& obj)
    : _device_id (obj._device_id)
    , _context (obj._context)
    , _cmd_queue (obj._cmd_queue)
    , _program (obj._program)
    , _kernels (obj._kernels)
    , _buffers (obj._buffers)
    , _next_key (obj._next_key)
    , _src (obj._src) {}

mpoi::~mpoi () { _cleanup_opencl(); }

mpoi&
mpoi::operator= (const mpoi& obj) {
    _next_key = 0;
    _src      = obj._src;

    _setup_opencl();

    if (_src != "") {
        build_program (_src);
    }

    return *this;
}

void
mpoi::_setup_opencl () {
    cl_uint num_platforms;
    cl_int  err = clGetPlatformIDs (0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms < 1) {
        std::cerr << "Failed to find any OpenCL platforms.\n";
        exit (1);
    }

    // cl_platform_id* platformIDs = (cl_platform_id*)new cl_platform_id[num_platforms];
    auto platformIDs = std::make_unique<cl_platform_id[]> (num_platforms);
    err              = clGetPlatformIDs (num_platforms, platformIDs.get(), NULL);

    if (err != CL_SUCCESS) {
        std::cerr << "Failed to find any OpenCL platforms.\n";
        exit (1);
    }
    std::cout << num_platforms << " OpenCL platform(s) found.\n";

    bool    device_found              = false;
    cl_uint current_max_compute_units = 0;
    cl_uint max_compute_units         = 0;

    for (cl_uint i = 0; i != num_platforms; i++) {
        std::cout << "Platform # " << i << std::endl;
        cl_uint num_devices;
        err = clGetDeviceIDs (platformIDs[i], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
        if (num_devices < 1) {
            std::cerr << "No GPU devices found for platform " << platformIDs[i] << std::endl;
        } else {
            device_found = true;
            std::cout << num_devices << " GPU device(s) found for platform " << platformIDs[i]
                      << std::endl;
            auto deviceIDs = std::make_unique<cl_device_id[]> (num_devices);
            for (cl_uint j = 0; j != num_devices; j++) {
                err = clGetDeviceIDs (platformIDs[i], CL_DEVICE_TYPE_GPU, 1, &deviceIDs[j], NULL);
                cl_uint device_vendor_id;
                err = clGetDeviceInfo (
                    deviceIDs[j], CL_DEVICE_VENDOR_ID, sizeof (cl_uint), &device_vendor_id, NULL
                );
                std::cout << "Device vendor ID: " << device_vendor_id << std::endl;
                err = clGetDeviceInfo (
                    deviceIDs[j],
                    CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof (cl_uint),
                    &max_compute_units,
                    NULL
                );
                std::cout << "Device has " << max_compute_units << " compute units.\n";
                if (max_compute_units > current_max_compute_units) {
                    current_max_compute_units = max_compute_units;
                    _device_id                = deviceIDs[j];
                }
            }
        }
    }

    if (!device_found) {
        std::cerr << "No OpenCL GPU devices found through all the platforms.\n";
        exit (1);
    }

    _context = clCreateContext (NULL, 1, &_device_id, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error in creating a context.\n";
        exit (1);
    }
    _cmd_queue = clCreateCommandQueue (_context, _device_id, 0, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error in creating a command queue.\n";
        exit (1);
    }
    _program = NULL;
}

void
mpoi::_cleanup_opencl () {
    clFlush (_cmd_queue);
    clFinish (_cmd_queue);
    for (std::size_t i = 0; i != _kernels.size(); i++) {
        clReleaseKernel (_kernels[i]);
    }
    if (!_program) {
        clReleaseProgram (_program);
    }
    clReleaseCommandQueue (_cmd_queue);
    clReleaseContext (_context);
}

void
mpoi::build_program (const std::string& src_file) {
    std::ifstream in (src_file);
    if (!in.is_open()) {
        std::cerr << "OpenCL program source not found: " << src_file << std::endl;
        exit (1);
    }
    std::string       src ((std::istreambuf_iterator<char> (in)), std::istreambuf_iterator<char>());
    const char*       src_string = src.c_str();
    const std::size_t src_length = src.length();
    cl_int            err;

    _program = clCreateProgramWithSource (
        _context, 1, (const char**)&src_string, (const std::size_t*)&src_length, &err
    );
    if (err != CL_SUCCESS) {
        std::cerr << "Error in creating a program.\n";
    }

    err = clBuildProgram (_program, 1, &_device_id, NULL, NULL, NULL);

    if (err != CL_SUCCESS) {
        std::cerr << "Error in building a program.\n";
        cl_build_status build_status;

        clGetProgramBuildInfo (
            _program,
            _device_id,
            CL_PROGRAM_BUILD_STATUS,
            sizeof (cl_build_status),
            &build_status,
            NULL
        );

        if (build_status != CL_SUCCESS) {
            std::size_t ret_val_size;
            clGetProgramBuildInfo (
                _program, _device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size
            );

            auto build_log = std::make_unique<char[]> (ret_val_size + 1);

            clGetProgramBuildInfo (
                _program, _device_id, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log.get(), NULL
            );

            build_log[ret_val_size] = '\0';

            std::cerr << "BUILD LOG: " << build_log << std::endl;
        }
    }
}

std::size_t
mpoi::create_kernel (const std::string& name) {
    std::size_t id = _kernels.size();
    cl_int      err;
    _kernels.push_back (clCreateKernel (_program, name.c_str(), &err));
    return id;
}

void
mpoi::display_platform_info () const {
    cl_uint num_platforms;
    cl_int  err = clGetPlatformIDs (0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms < 1) {
        std::cerr << "Failed to find any OpenCL platforms.\n";
        return;
    }

    auto platformIDs = std::make_unique<cl_platform_id[]> (num_platforms);

    err = clGetPlatformIDs (num_platforms, platformIDs.get(), NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to find any OpenCL platforms.\n";
        return;
    }
    std::cout << "Number of platforms: \t" << num_platforms << std::endl;

    for (cl_uint i = 0; i != num_platforms; i++) {
        _display_platform_info (platformIDs[i], CL_PLATFORM_PROFILE, "CL_PLATFORM_PROFILE   ");
        _display_platform_info (platformIDs[i], CL_PLATFORM_VERSION, "CL_PLATFORM_VERSION   ");
        _display_platform_info (platformIDs[i], CL_PLATFORM_VENDOR, "CL_PLATFORM_VENDOR    ");
        _display_platform_info (platformIDs[i], CL_PLATFORM_EXTENSIONS, "CL_PLATFORM_EXTENSIONS");
    }
}

void
mpoi::_display_platform_info (cl_platform_id id, cl_platform_info name, std::string str) const {
    std::size_t param_value_size;
    cl_int      err = clGetPlatformInfo (id, name, 0, NULL, &param_value_size);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to find OpenCL platform " << str << ".\n";
        return;
    }

    auto info = std::make_unique<char[]> (param_value_size);
    err       = clGetPlatformInfo (id, name, param_value_size, info.get(), NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to find OpenCL platform " << str << ".\n";
        return;
    }
    std::cout << "\t" << str << ":\t" << info << std::endl;
}

std::size_t
mpoi::create_buffer (mpoi::buffer_property bp, const std::size_t sz) {

    cl_int err;
    cl_mem buffer       = clCreateBuffer (_context, bp, sz, NULL, &err);
    _buffers[_next_key] = buffer;
    _next_key++;
    return _next_key - 1;
}

void
mpoi::release_buffer (const std::size_t id) {
    if (_buffers[id] != NULL) {
        clReleaseMemObject (_buffers[id]);
        _buffers[id] = NULL;
    }
}

void
mpoi::enqueue_write_buffer (const std::size_t id, const std::size_t size, const void* mem) {
    if (_buffers[id] != NULL) {
        cl_int err =
            clEnqueueWriteBuffer (_cmd_queue, _buffers[id], CL_TRUE, 0, size, mem, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            std::cerr << "Error in enqueuing a write buffer.\n";
            return;
        }
    }
}

void
mpoi::enqueue_read_buffer (const std::size_t id, const std::size_t size, void* mem) {
    if (_buffers[id] != NULL) {
        cl_int err =
            clEnqueueReadBuffer (_cmd_queue, _buffers[id], CL_TRUE, 0, size, mem, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            std::cerr << "Error in enqueuing a read buffer.\n";
            return;
        }
    }
}

void
mpoi::set_kernel_argument (
    const std::size_t kernel_id,
    const std::size_t order,
    const std::size_t buffer_id
) {
    if ((_kernels[kernel_id] != NULL) && (_buffers[buffer_id] != NULL)) {
        cl_int err = clSetKernelArg (
            _kernels[kernel_id],
            static_cast<cl_uint> (order),
            sizeof (cl_mem),
            (void*)&(_buffers[buffer_id])
        );
        if (err != CL_SUCCESS) {
            std::cerr << "Error in setting a kernel argument!\n";
            return;
        }
    }
}

void
mpoi::set_kernel_argument (
    const std::size_t id,
    const std::size_t order,
    const std::size_t size,
    void*             mem
) {
    if (_kernels[id] != NULL) {
        cl_int err = clSetKernelArg (_kernels[id], static_cast<cl_uint> (order), size, mem);

        if (err != CL_SUCCESS) {
            std::cerr << "Error in setting a kernel argument!\n";
            return;
        }
    }
}

void
mpoi::enqueue_data_parallel_kernel (
    const std::size_t id,
    std::size_t       num_global_items,
    std::size_t       num_local_items
) {
    if (_kernels[id] != NULL) {
        for (; num_local_items != 1; num_local_items--) {
            if (num_global_items % num_local_items == 0) break;
        }
        std::cout << "Local item size = " << num_local_items << std::endl;
        cl_int err = clEnqueueNDRangeKernel (
            _cmd_queue, _kernels[id], 1, NULL, &num_global_items, &num_local_items, 0, NULL, NULL
        );
        if (err != CL_SUCCESS) {
            std::cerr << "Error in enqueuing nd range kernel.\n";
            return;
        }
    }
}
