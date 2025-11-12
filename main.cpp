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

float frand() {
  return (float)rand() / RAND_MAX * 2 - 1;
} // 这个函数应该没有什么可以优化的地方

// 这个结构体可能可以优化成SOA
struct Star {
  float px, py, pz;
  float vx, vy, vz;
  float mass;
};

std::vector<Star> stars;

/*
  1. 可以使用#pragma omp simd
  从而允许编译器忽略可能存在的数据依赖（包含指针重叠），从而鼓励（而不是强制）编译器进行向量化
  2. 小循环体可以使用#pragma
  unroll，因为stars的大小是确定的，为48，刚好是2的N次方，我打算设定展开因子为4
*/
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

// 这里可以大量优化
/*
  1.
  非const引用/指针可以加上__restrict表示不会出现指针/引用重叠，从而允许编译器做激进的SIMD优化
  2. 使用std::sqrt()而不是C语言的遗产sqrt()
  3. 不要重复计算循环不变量 other.mass * G * dt / d2
  ，把他挪到外面去（用一个变量提前算好并储存）
  4. 可以使用#pragma omp simd
  从而允许编译器忽略可能存在的数据依赖（包含指针重叠），从而鼓励（而不是强制）编译器进行向量化
  5. 小循环体可以使用#pragma
  unroll，因为stars的大小是确定的，为48，刚好是2的N次方，我打算设定展开因子为4
*/
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

/*
  1.
  非const引用/指针可以加上__restrict表示不会出现指针/引用重叠，从而允许编译器做激进的SIMD优化
  2. 小循环体可以使用#pragma
  unroll，因为stars的大小是确定的，为48，刚好是2的N次方，我打算设定展开因子为4
*/
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
    // 这里可以优化，编译器能看到step()的实现，在高优化级别下可能会自动内联以消除函数调用开销，
    // 但由于step()函数包含嵌套循环且较为复杂，最终是否内联取决于编译器的内联阈值和代码膨胀考量
    for (int i = 0; i < 100000; i++)
      step();
  });
  printf("Final energy: %f\n", calc());
  printf("Time elapsed: %ld ms\n", dt);
  return 0;
}
