#include <deal.II/base/tensor.h>

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
    , strong_jacobian(n_dofs, n_dofs)
    , cell_rhs(n_dofs)
    , strong_residual(n_dofs)
  {}

  void
  reset_matrix_and_rhs()
  {
    cell_matrix     = 0;
    cell_rhs        = 0;
    strong_jacobian = 0;
    strong_residual = 0;
  }

  std::vector<Tensor<1, dim>> &
  get_solution_gradients()
  {
    return old_solution_gradients;
  }

  std::vector<double> &
  get_solution_laplacians()
  {
    return old_solution_laplacians;
  }

  std::vector<double> &
  get_JxW()
  {
    return JxW;
  }

  std::vector<std::vector<Tensor<2, dim>>> &
  get_hess_phi_u()
  {
    return hess_phi_u;
  }

  std::vector<std::vector<double>> &
  get_laplacian_phi_u()
  {
    return laplacian_phi_u;
  }

  std::vector<std::vector<double>> &
  get_phi_u()
  {
    return phi_u;
  }

  std::vector<std::vector<Tensor<1, dim>>> &
  get_grad_phi_u()
  {
    return grad_phi_u;
  }

  unsigned int
  get_n_q_points()
  {
    return n_q_points;
  }

  unsigned int
  get_n_dofs()
  {
    return n_dofs;
  }

  FullMatrix<double> &
  get_cell_matrix()
  {
    return cell_matrix;
  }

  Vector<double> &
  get_cell_rhs()
  {
    return cell_rhs;
  }

private:
  const unsigned int  n_q_points;
  const unsigned int  n_dofs;
  std::vector<double> JxW;

  std::vector<Tensor<1, dim>> old_solution_gradients;
  std::vector<double>         old_solution_laplacians;

  std::vector<std::vector<Tensor<2, dim>>> hess_phi_u;
  std::vector<std::vector<double>>         laplacian_phi_u;
  std::vector<std::vector<double>>         phi_u;
  std::vector<std::vector<Tensor<1, dim>>> grad_phi_u;

  FullMatrix<double> cell_matrix;
  FullMatrix<double> strong_jacobian;

  Vector<double> cell_rhs;
  Vector<double> strong_residual;
};

#endif
