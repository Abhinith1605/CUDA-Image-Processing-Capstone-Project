# GPU Image Filtering & Thermal Visualization (CUDA + OpenCV)

This project is my **Capstone submission** for the *CUDA at Scale for the Enterprise* Specialization.  
It demonstrates **high-performance GPU image processing** using custom CUDA kernels, OpenCV for I/O, and a complete command-line interface for reproducible experiments.

The final pipeline performs:

- **Grayscale conversion (CUDA)**
- **GPU min/max reduction (CUDA parallel reduction)**
- **Dynamic normalization to [0,1] range**
- **Custom thermal colormap (CUDA LUT mapping)**
- **High-resolution image output (OpenCV)**

The output is a **striking thermal/FLIR-style visualization**, ideal for presentations and demonstrating GPU acceleration.

---

## **Project Objectives**

### Demonstrate understanding of:
- CUDA kernel programming  
- Memory management (global, shared, coalesced access)  
- Parallel reductions  
- Per-pixel operations in 2D grids  
- Interfacing CUDA and OpenCV  

### Produce:
- A complete GitHub-ready codebase  
- A working executable  
- Before/after visual outputs  
- A 10-slide presentation (PDF)  
- Proof of GPU execution  
- Clean README documentation  

---

## **Build & Run Instructions (Windows, Visual Studio + CUDA Toolkit)**

### **1. Open "x64 Native Tools Command Prompt for VS 2022"**
This is required so that MSVC + NVCC are both available.

### **2. Navigate to your project**
```cmd
cd /d C:\Users\abhin\OneDrive\Desktop\CUDA-Image-Processing-Capstone-Project
```

### **3. Configure and generate build system with CMake**
```cmd
cmake -S . -B build -DOpenCV_DIR="C:/Users/abhin/Downloads/opencv/build"
```

### **4. Build the project**
```cmd
cmake --build build --config Release
```

### **5. Run the GPU executable**
```cmd
cd build/Release
GPU_Image_Filter.exe ..\..\input\input.jpg ..\..\output\output.png
```

- **NOTE:** The paths used here are according to my pc if you want to run the project just delete output folder and run the bellow command in your PC while in the Release directory. Which is Root Project Folder --> build --> Release
```cmd
GPU_Image_Filter.exe ..\..\input\input.jpg ..\..\output\output.png
```

---

## **CUDA Pipeline Overview**

### 1️⃣ **Convert BGR → Grayscale (CUDA kernel)**  
Formula:  
```
0.299 R + 0.587 G + 0.114 B
```

### 2️⃣ **Find min/max using shared memory reduction**  
Reduces **1 million+ pixels** in milliseconds.

### 3️⃣ **Normalize grayscale**  
```
norm = (value - min) / (max - min)
```

### 4️⃣ **Apply a thermal/FLIR-like colormap (LUT)**  
Each pixel is mapped to an RGB color based on intensity.

### 5️⃣ **Write final output using OpenCV**

---

## **Example Output**

### Input  
A normal photograph (Lena.jpg is what i used sourced from: https://github.com/opencv/opencv/tree/master)

### Output  
A **full thermal-style GPU visualization**, similar to FLIR or scientific imaging (Will be showin in Proof of Execution.)

---

## **Dependencies**

### Required:
- CUDA Toolkit ≥ 11.0  
- Visual Studio 2019 or 2022  
- OpenCV 4.x (Windows prebuilt package)  
