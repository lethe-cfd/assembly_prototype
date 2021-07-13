#include <deal.II/base/tensor.h>

#include <deal.II/lac/vector.h>

#include <vector>


using namespace dealii;

#ifndef scratch_h
#  define scratch_h

template <int dim>
class MinimalSurfaceScratch
{
public:
  MinimalSurfaceScratch(const unsigned int n_q_points,
                        const unsigned int n_dofs)
    : n_q_points(n_q_points)
    , n_dofs(n_dofs)
    , JxW(n_q_points)
    , old_solution_gradients(n_q_points)
    , old_solution_laplacians(n_q_points)
    , hess_phi_u(n_q_points, std::vector<Tensor<2, dim>>(n_dofs))
    , laplacian_phi_u(n_q_points, std::vector<double>(n_dofs))
    , phi_u(n_q_points, std::vector<double>(n_dofs))
    , grad_phi_u(n_q_points, std::vector<Tensor<1, dim>>(n_dofs))
    , cell_matrix(n_dofs, n_dofs)
    , cell_rhs(n_dofs)
    , strong_residual(n_q_points)
    , strong_jacobian(n_q_points, Vector<double>(n_dofs))
  {}

  void
  reset_matrix_and_rhs()
  {
    cell_matrix = 0;
    cell_rhs    = 0;
    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        strong_jacobian[q] = 0;
        strong_residual[q] = 0;
      }
  }

  const unsigned int  n_q_points;
  const unsigned int  n_dofs;
  double              cell_size;
  std::vector<double> JxW;

  std::vector<Tensor<1, dim>> old_solution_gradients;
  std::vector<double>         old_solution_laplacians;

  std::vector<std::vector<Tensor<2, dim>>> hess_phi_u;
  std::vector<std::vector<double>>         laplacian_phi_u;
  std::vector<std::vector<double>>         phi_u;
  std::vector<std::vector<Tensor<1, dim>>> grad_phi_u;

  FullMatrix<double> cell_matrix;



  Vector<double>              cell_rhs;
  Vector<double>              strong_residual;
  std::vector<Vector<double>> strong_jacobian;
};

#endif
