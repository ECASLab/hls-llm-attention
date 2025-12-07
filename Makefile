# ===========================================================
#              Makefile – Solo reporte HLS (Ruta B)
# ===========================================================

ROOT_DIR=$(realpath $(dir $(lastword $(MAKEFILE_LIST))))
TARGET := hw
PLATFORM ?= xilinx_u55c_gen3x16_xdma_3_202210_1

HLS_FILES := sdpa.cpp
HLS_FILES_NAMES := sdpa

TEMP_DIR := ./tmp.$(TARGET)
BUILD_DIR := ./build.$(TARGET)

# ================== NUEVO: Reporte HLS ======================
.PHONY: csynth
csynth: check_platform $(HLS_KERNEL_FILES)
	@echo "Reporte HLS generado (buscalo con: find $(TEMP_DIR) -name '*csynth.rpt')"

# ============================================================
# Packaging (no se usa en este flujo, pero se deja)
PACKAGE_OUT = ./package.$(TARGET)
LINK_OUTPUT := $(BUILD_DIR)/kernels.link.xclbin
XCL_BIN := $(PACKAGE_OUT)/kernels.xclbin
HLS_KERNEL_FILES := $(addprefix $(TEMP_DIR)/,$(HLS_FILES:.cpp=.xo))
# ============================================================

VPP_PFLAGS :=
VPP_LDFLAGS :=
VPP_FLAGS += --save-temps --jobs 6
VPP_FLAGS += -IHW -IHW/common -Imodules -Imodules/include
VPP_FLAGS += -DUSE_FLOAT32

KERNEL_FREQ := 250
RMDIR = rm -rf

.PHONY: all clean cleanall check_platform hls-build
all: csynth

check_platform:
ifndef PLATFORM
	$(error PLATFORM not set. Please set the PLATFORM properly and rerun. Run "make help" for more details.)
endif

# Compilación HLS (fase csynth). 
# Genera el .xo y el reporte HLS sin link ni empaquetado.
$(TEMP_DIR)/%.xo: %.cpp
	mkdir -p $(TEMP_DIR)
	v++ -c $(VPP_FLAGS) \
	    -t $(TARGET) \
	    --platform $(PLATFORM) \
	    -k $(<:.cpp=) \
	    --temp_dir $(TEMP_DIR) \
	    --kernel_frequency $(KERNEL_FREQ) \
	    -I'$(<D)' \
	    -o '$@' '$<'

# Limpieza ligera
clean:
	$(RMDIR) $(TEMP_DIR)
	$(RMDIR) *.log *.jou *.rpt *.csv

cleanall: clean
	$(RMDIR) build_dir* sd_card* package.* build.*
	$(RMDIR) _x* *xclbin.run_summary qemu-memory-* emulation *.xclbin
