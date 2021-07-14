#include <deal.II/base/tensor.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/vector.h>

#include <vector>


using namespace dealii;

#ifndef scratch_h
#  define scratch_h


template <int dim>
class MinimalSurfaceAssemblyCacheData
{
public:
  MinimalSurfaceAssemblyCacheData(const unsigned int n_q_points,
                                  const unsigned int n_dofs)
    : n_q_points(n_q_points)
    , n_dofs(n_dofs)
    , JxW(n_q_points)
    , old_solution_gradients(n_q_points)
    , old_solution_laplacians(n_q_points)
    , hess_phi_u(n_q_points, std::vector<Tensor<2, dim>>(n_dofs))
    , laplacian_phi_u(n_q_points, std::vector<double>(n_dofs))
    , phi_u(n_q_points, std::vector<double>(n_dofs))
    , grad_phi_u(n_q_points, std::vector<Tensor<1, dim>>(n_dofs)){};

  unsigned int        n_q_points;
  unsigned int        n_dofs;
  double              cell_size;
  std::vector<double> JxW;

  std::vector<Tensor<1, dim>> old_solution_gradients;
  std::vector<double>         old_solution_laplacians;

  std::vector<std::vector<Tensor<2, dim>>> hess_phi_u;
  std::vector<std::vector<double>>         laplacian_phi_u;
  std::vector<std::vector<double>>         phi_u;
  std::vector<std::vector<Tensor<1, dim>>> grad_phi_u;
};

template <int dim>
class MinimalSurfaceAssemblyScratchData
{
public:
  MinimalSurfaceAssemblyScratchData(const FE_Q<dim> &fe,
                                    QGauss<dim> &    quadrature_formula,
                                    MappingQ1<dim> & mapping)
    : fe_values(mapping,
                fe,
                quadrature_formula,
                update_values | update_gradients | update_quadrature_points |
                  update_JxW_values | update_hessians)
    , cache_data(fe_values.get_quadrature().size(),
                 fe_values.get_fe().n_dofs_per_cell()){};

  MinimalSurfaceAssemblyScratchData(const MinimalSurfaceAssemblyScratchData &sd)
    : fe_values(sd.fe_values.get_mapping(),
                sd.fe_values.get_fe(),
                sd.fe_values.get_quadrature(),
                update_values | update_gradients | update_quadrature_points |
                  update_JxW_values | update_hessians)
    , cache_data(fe_values.get_quadrature().size(),
                 fe_values.get_fe().n_dofs_per_cell()){};

  FEValues<dim>                        fe_values;
  MinimalSurfaceAssemblyCacheData<dim> cache_data;
};



class MinimalSurfaceAssemblyCopyData
{
public:
  unsigned int                         n_dofs;
  FullMatrix<double>                   cell_matrix;
  Vector<double>                       cell_rhs;
  Vector<double>                       strong_residual;
  std::vector<Vector<double>>          strong_jacobian;
  std::vector<types::global_dof_index> local_dof_indices;
};

#endif
