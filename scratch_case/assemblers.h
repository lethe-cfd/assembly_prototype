#include "scratch.h"

#ifndef assembler_h
#  define assembler_h

template <int dim>
class AssemblerBase
{
public:
  virtual void
  assemble(MinimalSurfaceScratch<dim> &scratch) = 0;
};


template <int dim>
class AssemblerClassic : public AssemblerBase<dim>
{
public:
  virtual void
  assemble(MinimalSurfaceScratch<dim> &scratch) override
  {
    const auto &phi_u           = scratch.get_phi_u();
    const auto &grad_phi_u      = scratch.get_grad_phi_u();
    const auto &laplacian_phi_u = scratch.get_laplacian_phi_u();

    const auto &old_solution_laplacians = scratch.get_solution_laplacians();
    const auto &old_solution_gradients  = scratch.get_solution_gradients();

    const auto &       JxW_vec    = scratch.get_JxW();
    const unsigned int n_q_points = scratch.get_n_q_points();
    const unsigned int n_dofs     = scratch.get_n_dofs();

    auto &cell_matrix = scratch.get_cell_matrix();
    auto &cell_rhs    = scratch.get_cell_rhs();

    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        const double coeff = 1.0 / std::sqrt(1 + old_solution_gradients[q] *
                                                   old_solution_gradients[q]);

        const double tau =
          0.; // 1. / std::sqrt(9 * std::pow(4 * coeff / (h * h), 2));

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
                  tau * phi_u[q][i] * coeff * laplacian_phi_u[q][j] * JxW;
              }

            cell_rhs(i) -=
              (grad_phi_u[q][i] * old_solution_gradients[q] * coeff * JxW);

            // Pseudo GLS term
            cell_rhs(i) -=
              tau * phi_u[q][i] * coeff * old_solution_laplacians[q] * JxW;
          }
      }
  };
};

#endif
