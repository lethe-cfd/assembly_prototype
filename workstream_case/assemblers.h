#include "scratch.h"

#ifndef assembler_h
#  define assembler_h

template <int dim>
class AssemblerBase
{
public:
  virtual void
  assemble(MinimalSurfaceAssemblyCacheData<dim> &cache_data,
           MinimalSurfaceAssemblyCopyData &      copy_data) = 0;
};


template <int dim>
class AssemblerMain : public AssemblerBase<dim>
{
public:
  virtual void
  assemble(MinimalSurfaceAssemblyCacheData<dim> &cache_data,
           MinimalSurfaceAssemblyCopyData &      copy_data) override
  {
    const auto &phi_u      = cache_data.phi_u;
    const auto &grad_phi_u = cache_data.grad_phi_u;

    const auto &old_solution_gradients = cache_data.old_solution_gradients;

    const auto &       JxW_vec    = cache_data.JxW;
    const unsigned int n_q_points = cache_data.n_q_points;
    const unsigned int n_dofs     = cache_data.n_dofs;

    auto &strong_residual = copy_data.strong_residual;
    auto &strong_jacobian = copy_data.strong_jacobian;

    auto &cell_matrix = copy_data.cell_matrix;
    auto &cell_rhs    = copy_data.cell_rhs;

    const double h = cache_data.cell_size;


    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        const double coeff = 1.0 / std::sqrt(1 + old_solution_gradients[q] *
                                                   old_solution_gradients[q]);

        const double tau = 1. / std::sqrt(9 * std::pow(4 * coeff / (h * h), 2));

        const double JxW = JxW_vec[q];

        for (unsigned int i = 0; i < n_dofs; ++i)
          {
            for (unsigned int j = 0; j < n_dofs; ++j)
              {
                cell_matrix(i, j) +=
                  (((grad_phi_u[q][i]                // ((\nabla \phi_i
                     * coeff                         //   * a_n
                     * grad_phi_u[q][j])             //   * \nabla \phi_j)
                    -                                //  -
                    (grad_phi_u[q][i]                //  (\nabla \phi_i
                     * coeff * coeff * coeff         //   * a_n^3
                     * (grad_phi_u[q][j]             //   * (\nabla \phi_j
                        * old_solution_gradients[q]) //      * \nabla u_n)
                     * old_solution_gradients[q]))   //   * \nabla u_n)))
                   * JxW);                           // * dx

                // Pseudo GLS term
                cell_matrix(i, j) +=
                  tau * phi_u[q][i] * strong_jacobian[q][j] * JxW;
              }

            cell_rhs(i) -=
              (grad_phi_u[q][i] * old_solution_gradients[q] * coeff * JxW);

            // Pseudo GLS term
            cell_rhs(i) -= tau * phi_u[q][i] * strong_residual[q] * JxW;
          }
      }
  };
};


template <int dim>
class AssemblerStabilization : public AssemblerBase<dim>
{
public:
  virtual void
  assemble(MinimalSurfaceAssemblyCacheData<dim> &cache_data,
           MinimalSurfaceAssemblyCopyData &      copy_data) override
  {
    const auto &phi_u           = cache_data.phi_u;
    const auto &laplacian_phi_u = cache_data.laplacian_phi_u;

    const auto &old_solution_laplacians = cache_data.old_solution_laplacians;
    const auto &old_solution_gradients  = cache_data.old_solution_gradients;

    const unsigned int n_q_points = cache_data.n_q_points;
    const unsigned int n_dofs     = cache_data.n_dofs;

    auto &strong_residual = copy_data.strong_residual;
    auto &strong_jacobian = copy_data.strong_jacobian;


    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        const double coeff = 1.0 / std::sqrt(1 + old_solution_gradients[q] *
                                                   old_solution_gradients[q]);

        for (unsigned int j = 0; j < n_dofs; ++j)
          {
            strong_jacobian[q][j] += coeff * laplacian_phi_u[q][j];
          }

        for (unsigned int i = 0; i < n_dofs; ++i)
          {
            strong_residual[q] +=
              phi_u[q][i] * coeff * old_solution_laplacians[q];
          }
      }
  };
};

#endif
