#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/cl2.hpp>

std::vector<cl::Platform> platforms;
std::vector<std::vector<cl::Device>> devices;
cl::Context context;
cl::CommandQueue commandQueue;
cl::Device device;

#define CH_IN 8
#define CH_DOWNMIX 2
#define T float

typedef std::chrono::high_resolution_clock Clock;


void getDevices()
{
    try { cl::Platform::get(&platforms); }
    catch (cl::Error& e) { std::cerr << "OpenCL get platform Error:" << e.what() << "(" << e.err() << ")" << std::endl; exit( e.err());}
    std::cout << "Found "<< platforms.size() << " platforms." << std::endl;
    int platID = -1;
    for (auto &p : platforms)
    {
        std::cout << "OpenCL platform found: " <<  p.getInfo<CL_PLATFORM_NAME>() << ", " << p.getInfo<CL_PLATFORM_VERSION>() << ", " << p.getInfo<CL_PLATFORM_VENDOR>() << std::endl;
        devices.emplace_back();
        platID += 1;
        auto& ds = devices.back();
        try { p.getDevices(CL_DEVICE_TYPE_ALL, &ds); }
        catch (cl::Error& e) {	std::cerr <<  "OpenCL getDeviced Error: " << e.what() << "("<< e.err() <<")"<< std::endl; exit(e.err());}
        int devID = -1;
        for (auto& d : ds)
        {
            std::cout << " (" << platID << "," << ++devID << ") Device found: " << d.getInfo<CL_DEVICE_NAME>() << ", " <<
                      d.getInfo<CL_DEVICE_VERSION>() << ", " << d.getInfo<CL_DEVICE_VENDOR>() << ", " <<
                      d.getInfo<CL_DRIVER_VERSION>() << std::endl;
        }
    }
}

void activateDevice(size_t platform, size_t devId)
{
    if (platform > platforms.size() || devId > devices.size())
    {
        throw std::runtime_error("Illegal platform/device ID");
    }
    device = devices[platform][devId];
    context = cl::Context(device);
    commandQueue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
    printf("OpenCL context and command queue created (refcounts: %d, %d)\n", context.getInfo<CL_CONTEXT_REFERENCE_COUNT>(), commandQueue.getInfo<CL_QUEUE_REFERENCE_COUNT>());
}

std::string loadFile(const std::string &fileName)
{
    try {
        std::ifstream file;
        file.open(fileName);
        std::stringstream ss;
        ss << file.rdbuf();
        file.close();
        std::cout << "Loaded file \"" << fileName << "\" containing "<< ss.str().size() << " bytes" << std::endl;
        return ss.str();
    }
    catch (std::exception& e)
    {
        std::cerr << "Cound not open file \"" << fileName << "\": " << e.what() << std::endl;
        exit(-1);
    }
}

void writeFile(const std::string & fileName, const std::string& content)
{
    try {
        std::ofstream file(fileName);
        file << content;
        file.close();
    }
    catch (std::exception& e)
    {
        std::cerr << "Cound not open file \"" << fileName << "\": " << e.what() << std::endl;
        exit(-1);
    }
}

std::vector<cl::Kernel> createKernel(const std::string& code)
{
    cl::Program program{ context, code };
    try {
        program.build();
        std::vector<cl::Kernel> kernels;
        program.createKernels(&kernels);
        std::cout << "Created " << kernels.size() << " kernels" << std::endl;
        for (cl::Kernel& k : kernels) { std::cout << "Created kernel \"" << k.getInfo<CL_KERNEL_FUNCTION_NAME>() << "\"" << std::endl; }
        return kernels;
    }
    catch (cl::Error& e)
    {
        std::string fileName = "kernels\\buildlog.txt";
        std::cerr << " Error creating OpenCL kernel: " << e.what() << "(" << e.err() << ") " << fileName.c_str() << std::endl;
        if (e.err() == CL_BUILD_PROGRAM_FAILURE)
        {
            writeFile(fileName,program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));
        }
        exit(e.err());
    }
}

template <typename Rep, typename Period>
std::ostream& operator << (std::ostream& os, const std::chrono::duration<Rep,Period>& dur)
{
    os << std::chrono::duration_cast<std::chrono::microseconds>(dur).count() << "us";
    return os;
}

int main() {
    getDevices();

    size_t p, d;
    std::cout << "Choose platform: ";
    std::cin >> p;
    std::cout << "Choose device: ";
    std::cin >> d;
    activateDevice(p, d);

    size_t bufferLength = 48 * 11;
    T* frameBufferIn =  (T*)malloc(bufferLength*CH_IN * sizeof(T));
    T* clResult = (T*)malloc(bufferLength*CH_DOWNMIX * sizeof(T));

    std::string dmstr = loadFile("kernels\\DownMix.cl");
    std::vector<cl::Kernel> kernels = createKernel(dmstr);
    cl::Kernel OCLdownMix = kernels[0];
    cl::Buffer clIn{ context, CL_MEM_READ_WRITE, bufferLength * CH_IN * sizeof(T)};
    cl::Buffer clOut{ context, CL_MEM_READ_WRITE, bufferLength * CH_DOWNMIX * sizeof(T)};

    int run = 0;



    for (int i = 0; i < 10; ++i) {
        cl::Event copyToEvent;
        cl::Event copyFromEvent;
        cl::Event kernelEvent;
        try {
            auto t1 = Clock::now();
            commandQueue.enqueueWriteBuffer(clIn, CL_FALSE, 0, 10 * 48 * sizeof(float), frameBufferIn, nullptr, &copyToEvent);
            OCLdownMix.setArg(0,clIn);
            OCLdownMix.setArg(1,clOut);
            OCLdownMix.setArg(2,(unsigned int)480);
            commandQueue.enqueueNDRangeKernel(OCLdownMix, cl::NullRange, cl::NDRange(480), cl::NDRange(48), nullptr, &kernelEvent);
            commandQueue.enqueueReadBuffer(clOut, CL_FALSE, 0, 10 * 48 * sizeof(float), clResult, nullptr, &copyFromEvent);
            auto t2 = Clock::now();
            commandQueue.finish();
            auto t3 = Clock::now();
            cl_ulong copyToTime = copyToEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                                  copyToEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong kernelTime = kernelEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                                  kernelEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong copyFromTime = copyFromEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                                  copyFromEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            std::cout << "Enqueue: " << t2 - t1 << ", Total: " << t3 - t1 << ", GPU: " << (copyToTime+kernelTime+copyFromTime) / 1000.0 << "us"<< std::endl;
        } catch (std::exception& e)
        {
            std::cerr << "Caught exception: " << e.what() << std::endl;
            exit(-2);
        }

        run++;

    }

    return 0;
}
