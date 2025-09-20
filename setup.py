# setup.py (isolation-safe)

import os
import glob
import logging
import subprocess
from setuptools import setup

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s - %(message)s")

PACKAGE_NAME = "torchsdf"
LICENSE = "Apache-2.0"
cwd = os.path.dirname(os.path.abspath(__file__))

version_txt = os.path.join(cwd, "version.txt")

def read_version():
    # Try version.txt; fall back to env or default
    if os.path.exists(version_txt):
        with open(version_txt) as f:
            return f.readline().strip()
    return os.environ.get("TORCHSDF_VERSION", "0.0.0")

version = read_version()

def write_version_file():
    version_path = os.path.join(cwd, "torchsdf", "version.py")
    os.makedirs(os.path.dirname(version_path), exist_ok=True)
    with open(version_path, "w") as f:
        f.write(f"__version__ = '{version}'\n")

def write_version_file():
    version_path = os.path.join(cwd, PACKAGE_NAME, "version.py")
    with open(version_path, "w") as f:
        f.write(f"__version__ = '{version}'\n")

def get_cuda_bare_metal_version(cuda_dir="/usr/local/cuda"):
    raw_output = subprocess.check_output([os.path.join(cuda_dir, "bin/nvcc"), "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    major, minor = output[release_idx].split(".")
    return raw_output, major, minor[0]

def get_include_dirs():
    return []

def build_extensions():
    """
    Try to build C++/CUDA extensions using torch.utils.cpp_extension *if available*.
    Never import torch at module import time to keep PEP517 isolation happy.
    """
    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME
    except Exception as e:
        logger.warning("Torch/C++ extension toolchain not available during build (%s). "
                       "Building a pure-Python wheel without native ops.", e)
        return [], {}

    sources = glob.glob(f"{PACKAGE_NAME}/csrc/**/*.cpp", recursive=True)
    define_macros = []
    extra_compile_args = {"cxx": ["-O3"]}
    include_dirs = []

    use_cuda = torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1"
    if use_cuda:
        define_macros += [("WITH_CUDA", None), ("THRUST_IGNORE_CUB_VERSION_CHECK", None)]
        sources += glob.glob(f"{PACKAGE_NAME}/csrc/**/*.cu", recursive=True)
        extension_cls = CUDAExtension
        extra_compile_args.update({"nvcc": ["-O3", "-DWITH_CUDA", "-DTHRUST_IGNORE_CUB_VERSION_CHECK"]})
        include_dirs = get_include_dirs()

        # populate default arch list if cross compiling
        if not torch.cuda.is_available() and os.getenv("FORCE_CUDA", "0") == "1" and os.getenv("TORCH_CUDA_ARCH_LIST") is None:
            try:
                _, major, minor = get_cuda_bare_metal_version(CUDA_HOME or "/usr/local/cuda")
                if int(major) == 11:
                    os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;" + ("8.0" if int(minor) == 0 else "8.0;8.6")
                else:
                    os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5"
            except Exception:
                pass
    else:
        extension_cls = CppExtension

    ext = extension_cls(
        name=f"{PACKAGE_NAME}._C",
        sources=sources,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        include_dirs=include_dirs,
    )

    # Example of post-processing libraries if needed:
    # ext.libraries = ["cudart_static" if x == "cudart" else x for x in (ext.libraries or [])]

    cmdclass = {"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)}
    return [ext], cmdclass

if __name__ == "__main__":
    write_version_file()

    # Allow disabling native build explicitly (useful for CI / sdist): TORCHSDF_BUILD_EXT=0
    build_ext = os.getenv("TORCHSDF_BUILD_EXT", "1") == "1"
    ext_modules, cmdclass = ([], {}) if not build_ext else build_extensions()

    setup(
        name=PACKAGE_NAME,
        version=version,
        license=LICENSE,
        zip_safe=False,
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        # Let pyproject.toml control packages/data discovery (see next section)
    )
