#ifndef __MULTI_PROCESSING_OBJECT_INTERFACE_H_
#define __MULTI_PROCESSING_OBJECT_INTERFACE_H_

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <vector>

class mpoi {
  public:
    enum buffer_property {
        READ_ONLY  = CL_MEM_READ_ONLY,
        WRITE_ONLY = CL_MEM_WRITE_ONLY,
        READ_WRITE = CL_MEM_READ_WRITE
    };

  private:
    cl_device_id                  _device_id;
    cl_context                    _context;
    cl_command_queue              _cmd_queue;
    cl_program                    _program;
    std::vector<cl_kernel>        _kernels;
    std::map<std::size_t, cl_mem> _buffers;
    std::size_t                   _next_key;
    std::string                   _src;

  public:
    mpoi ();
    mpoi (const std::string&);
    mpoi (const mpoi&);
    virtual ~mpoi ();

    mpoi&
    operator= (const mpoi&);

    void
    build_program (const std::string&);

    std::size_t
    create_kernel (const std::string&);

    void
    display_platform_info () const;

    std::size_t
    create_buffer (mpoi::buffer_property, const std::size_t);

    void
    release_buffer (const std::size_t);

    void
    enqueue_write_buffer (const std::size_t, const std::size_t, const void*);

    void
    enqueue_read_buffer (const std::size_t, const std::size_t, void*);

    void
    set_kernel_argument (const std::size_t, const std::size_t, const std::size_t);

    void
    set_kernel_argument (const std::size_t, const std::size_t, const std::size_t, void*);

    void
    enqueue_data_parallel_kernel (const std::size_t, std::size_t, std::size_t);

  private:
    void
    _setup_opencl ();

    void
    _cleanup_opencl ();

    void _display_platform_info (cl_platform_id, cl_platform_info, std::string) const;
};

#endif
