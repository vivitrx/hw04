/*
老师希望你做的事情：

【作业目标】
优化这个多体引力求解器，获得尽可能高的性能加速倍数。

【评分规则】
- 加速倍数 = 基础分数（例如：50倍加速 = 50分）
- 详细解释优化原理可以获得额外乘法加分（例如：50倍 × 2 = 100分）
- 没有详细解释则分数为0（例如：1000倍 × 0 = 0分）

【限制条件】
❌ 不允许使用多线程并行
❌ 不允许做算法复杂度优化（不能改变O(n²)的基本复杂度）
✅ 可以针对编译器和平台优化（不要求跨平台）
✅ 可以使用xmmintrin.h进行SIMD优化

【优化方向建议】
1. SIMD向量化：使用SSE/AVX指令并行处理多个浮点运算
2. 数据结构优化：考虑AOS vs SOA的内存访问模式
3. 编译器优化：利用编译器内置函数和优化选项
4. 数学运算优化：减少重复计算，使用更高效的数学函数
5. 内存访问优化：提高缓存命中率，减少内存延迟

【性能测试】
- 基准测试：原始代码运行时间约为1339ms
- 需要在PR描述中记录优化前后的具体用时数据
- 需要详细解释每种优化技术的原理和效果

【注意事项】
- 保持计算结果的正确性（能量守恒）
- 代码需要有良好的注释说明优化思路
- 最终提交时需要包含详细的性能分析报告
*/

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>


float frand() { return (float)rand() / RAND_MAX * 2 - 1; }

struct Star {
  float px, py, pz;
  float vx, vy, vz;
  float mass;
};

std::vector<Star> stars;

void init() {
  for (int i = 0; i < 48; i++) {
    stars.push_back({
        frand(),
        frand(),
        frand(),
        frand(),
        frand(),
        frand(),
        frand() + 1,
    });
  }
}

float G = 0.001;
float eps = 0.001;
float dt = 0.01;

void step() {
  for (auto &star : stars) {
    for (auto &other : stars) {
      float dx = other.px - star.px;
      float dy = other.py - star.py;
      float dz = other.pz - star.pz;
      float d2 = dx * dx + dy * dy + dz * dz + eps * eps;
      d2 *= sqrt(d2);
      star.vx += dx * other.mass * G * dt / d2;
      star.vy += dy * other.mass * G * dt / d2;
      star.vz += dz * other.mass * G * dt / d2;
    }
  }
  for (auto &star : stars) {
    star.px += star.vx * dt;
    star.py += star.vy * dt;
    star.pz += star.vz * dt;
  }
}

float calc() {
  float energy = 0;
  for (auto &star : stars) {
    float v2 = star.vx * star.vx + star.vy * star.vy + star.vz * star.vz;
    energy += star.mass * v2 / 2;
    for (auto &other : stars) {
      float dx = other.px - star.px;
      float dy = other.py - star.py;
      float dz = other.pz - star.pz;
      float d2 = dx * dx + dy * dy + dz * dz + eps * eps;
      energy -= other.mass * star.mass * G / sqrt(d2) / 2;
    }
  }
  return energy;
}

template <class Func> long benchmark(Func const &func) {
  auto t0 = std::chrono::steady_clock::now();
  func();
  auto t1 = std::chrono::steady_clock::now();
  auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
  return dt.count();
}

int main() {
  init();
  printf("Initial energy: %f\n", calc());
  auto dt = benchmark([&] {
    for (int i = 0; i < 100000; i++)
      step();
  });
  printf("Final energy: %f\n", calc());
  printf("Time elapsed: %ld ms\n", dt);
  return 0;
}
