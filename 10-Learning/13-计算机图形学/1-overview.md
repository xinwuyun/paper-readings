Technical Challenges:
- Math of projections, curves, surfaces
- Physics of lighting and shading
- Representing / operating shape in 3D
- Animation /simulation

Course Topics
- Rasterization 光栅化
- Curves and meshes 曲线和曲面
- Ray Tracing
- Animation / Simulation 

## Rasterization 
把三维空间的几何形体显示在屏幕上
- Project **geometry primitives(3D triangles / polygons多边形)** onto the screen.
- Break projected primitives into **fragemnts(pixels)**
- Gold standard in Video Games (**Real-time applications**)
Real-time: 30 fps 60fps
otherwise offline.
![](../../08-Assets/Pasted%20image%2020231014170418.png)

## Curves and Meshes 
How to represent geometry in CG

如何保持物体拓扑结构，物体结构发生变化该如何表现。

![](../../08-Assets/Pasted%20image%2020231014170530.png)

## Ray Tracing 

Shoot rays from the camera though each pixel
- **Intersection交汇点** and **shading** 
- **Continue to bounce** the rays till they hit light sources

Gold standard in Animation / Movies (**Offline**)
 ![](../../08-Assets/Pasted%20image%2020231014170612.png)

## Computer Graphics vs Computer Vision

![](../../08-Assets/Pasted%20image%2020231014225659.png)

