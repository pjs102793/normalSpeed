#include "normal.hpp"
#include <omp.h>
#include <opencv2/imgproc/imgproc.hpp>

void accumBilateral(long delta, long i, long j, long *A, long *b, int threshold)
{
  long f = std::abs(delta) < threshold ? 1 : 0;

  const long fi = f * i;
  const long fj = f * j;

  A[0] += fi * i;
  A[1] += fi * j;
  A[3] += fj * j;
  b[0] += fi * delta;
  b[1] += fj * delta;
}

cv::Mat depthNormals(const cv::Mat &src, 
                 const float fx, 
                 const float fy,  
                 const int kernel_size, 
                 const int distance_threshold,
                 const int difference_threshold,
                 const bool norm_point_into_face)
{
  cv::Mat normal = cv::Mat::zeros(src.size(), CV_32FC4); 
  cv::Mat dst_x  = cv::Mat::zeros(src.size(), CV_32F);
  cv::Mat dst_y  = cv::Mat::zeros(src.size(), CV_32F);
  cv::Mat dst_z  = cv::Mat::zeros(src.size(), CV_32F);

  const unsigned short *lp_depth = src.ptr<ushort>(); 
  
  float *lp_x_normals = dst_x.ptr<float>(); 
  float *lp_y_normals = dst_y.ptr<float>(); 
  float *lp_z_normals = dst_z.ptr<float>(); 

  const int l_W = src.cols;
  const int l_H = src.rows;
  
  const int l_r = kernel_size;
  const int l_offset0 = -l_r - l_r * l_W;  
  const int l_offset1 = 0 - l_r * l_W;
  const int l_offset2 = +l_r - l_r * l_W;
  const int l_offset3 = -l_r;
  const int l_offset4 = +l_r;
  const int l_offset5 = -l_r + l_r * l_W;
  const int l_offset6 = 0 + l_r * l_W;
  const int l_offset7 = +l_r + l_r * l_W;

  #pragma omp parallel for
  for (int l_y = l_r; l_y < l_H - l_r - 1; ++l_y)
  {
    const unsigned short *lp_line = lp_depth + (l_y * l_W + l_r);
    float *lp_x_norm = lp_x_normals + (l_y * l_W + l_r);
    float *lp_y_norm = lp_y_normals + (l_y * l_W + l_r);
    float *lp_z_norm = lp_z_normals + (l_y * l_W + l_r);

    for (int l_x = l_r; l_x < l_W - l_r - 1; ++l_x)
    {
      long l_d = lp_line[0];
      
      if (l_d < distance_threshold)
      {
        long l_A[4];
        l_A[0] = l_A[1] = l_A[2] = l_A[3] = 0;
        long l_b[2];
        l_b[0] = l_b[1] = 0;
        accumBilateral(lp_line[l_offset0] - l_d, -l_r, -l_r, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset1] - l_d, 0, -l_r, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset2] - l_d, +l_r, -l_r, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset3] - l_d, -l_r, 0, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset4] - l_d, +l_r, 0, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset5] - l_d, -l_r, +l_r, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset6] - l_d, 0, +l_r, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset7] - l_d, +l_r, +l_r, l_A, l_b, difference_threshold);

        long l_det = l_A[0] * l_A[3] - l_A[1] * l_A[1];
        long l_ddx = l_A[3] * l_b[0] - l_A[1] * l_b[1];
        long l_ddy = -l_A[1] * l_b[0] + l_A[0] * l_b[1];

        float l_nx = static_cast<float>(fx * l_ddx);
        float l_ny = static_cast<float>(fy * l_ddy);
        float l_nz = static_cast<float>(-l_det * l_d);

        if(l_nz < 0 && norm_point_into_face)
        {
          l_nx *= -1;
          l_ny *= -1;
          l_nz *= -1;
        }

        float l_sqrt = sqrtf(l_nx * l_nx + l_ny * l_ny + l_nz * l_nz);

        if (l_sqrt > 0)
        {
          float l_norminv = 1.0f / l_sqrt;

          l_nx *= l_norminv;
          l_ny *= l_norminv;
          l_nz *= l_norminv;

          *lp_x_norm = l_nx;
          *lp_y_norm = l_ny;
          *lp_z_norm = l_nz;
        }
        else
        {
           *lp_x_norm = 0;
           *lp_y_norm = 0;
           *lp_z_norm = 0;
        }
      }
      else
      {
           *lp_x_norm = 0;
           *lp_y_norm = 0;
           *lp_z_norm = 0;
      }
      ++lp_line;
      ++lp_x_norm;
      ++lp_y_norm;
      ++lp_z_norm;
    }
  }
  
  cv::medianBlur(dst_x, dst_x, 3);
  cv::medianBlur(dst_y, dst_y, 3);
  cv::medianBlur(dst_z, dst_z, 3);

  std::vector<cv::Mat> mbgr(3); 
  mbgr[0] = dst_x;
  mbgr[1] = dst_y;
  mbgr[2] = dst_z;
  cv::merge(mbgr, normal);

  return normal;
}