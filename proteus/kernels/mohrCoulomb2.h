
inline void mohrCoulomb2(const double& phi, const double& psi, const double& c, const double* stress, //inputs
			double& f, double* df, double& g, double* r, double* dr) //outputs
{
  const int sXX=0,sYY=1,sZZ=2,sYZ=3,sZX=4,sXY=5;
  const double &sigma_x(stress[sXX]),
    &sigma_y(stress[sYY]),
    &sigma_z(stress[sZZ]),
    &sigma_yz(stress[sYZ]),
    &sigma_zx(stress[sZX]),
    &sigma_xy(stress[sXY]);
  const double eps(1.0e-8);
  double Df_sigma_m,Df_sigma_b,Df_theta,//f is function arg
    theta,Dtheta_arg,Dtheta_x,Dtheta_y,Dtheta_z,Dtheta_xy,Dtheta_yz,Dtheta_zx,
    sin_theta,cos_theta,
    sigma_m,Dsigma_m_x,Dsigma_m_y,Dsigma_m_z,
    sigma_b,sigma_b_star,Dsigma_b_x,Dsigma_b_y,Dsigma_b_z,Dsigma_b_xy,Dsigma_b_yz,Dsigma_b_zx,
    t,t_star,Dt_x,Dt_y,Dt_z,Dt_yz,Dt_zx,Dt_xy,
    s_x,Ds_x_x,Ds_x_y,Ds_x_z,
    s_y,Ds_y_x,Ds_y_y,Ds_y_z,
    s_z,Ds_z_x,Ds_z_y,Ds_z_z,
    J_3,DJ_3_x,DJ_3_y,DJ_3_z,DJ_3_xy,DJ_3_yz,DJ_3_zx,
    arg,arg_star,Darg_J_3,Darg_t,
    Dg_sigma_m,Dg_sigma_b,Dg_theta;//g is function arg
  
  sigma_m =  (sigma_x + sigma_y + sigma_z)/3.0; 
  Dsigma_m_x = 1.0/3.0;
  Dsigma_m_y = 1.0/3.0;
  Dsigma_m_z = 1.0/3.0;
  //sigma_b
  sigma_b =  sqrt(pow(sigma_x - sigma_y,2) + 
		  pow(sigma_y - sigma_z,2) + 
		  pow(sigma_z - sigma_x,2) + 
		  6.0*pow(sigma_xy,2) + 
		  6.0*pow(sigma_yz,2) + 
		  6.0*pow(sigma_zx,2))/ sqrt(2.0);
  sigma_b_star = sigma_b;
  if (sigma_b < eps)
    {
      sigma_b_star = eps;
    }
  Dsigma_b_x =  (2.0*sigma_x - sigma_y - sigma_z)/(2.0*sigma_b_star);
  Dsigma_b_y =  (2.0*sigma_y - sigma_x - sigma_z)/(2.0*sigma_b_star);
  Dsigma_b_z =  (2.0*sigma_z - sigma_y - sigma_x)/(2.0*sigma_b_star);
  Dsigma_b_xy = (3.0*sigma_xy)/sigma_b_star;
  Dsigma_b_yz = (3.0*sigma_yz)/sigma_b_star;
  Dsigma_b_zx = (3.0*sigma_zx)/sigma_b_star;
  
  //t
  t =  sigma_b*sqrt(2.0/3.0);
  t_star = t;
  if(t < eps)
    {
      t_star = eps;
    }
  Dt_x = Dsigma_b_x*sqrt(2.0/3.0);
  Dt_y = Dsigma_b_y*sqrt(2.0/3.0);
  Dt_z = Dsigma_b_z*sqrt(2.0/3.0);
  Dt_xy = Dsigma_b_xy*sqrt(2.0/3.0);
  Dt_yz = Dsigma_b_yz*sqrt(2.0/3.0);
  Dt_zx = Dsigma_b_zx*sqrt(2.0/3.0);

  //J_3
  s_x = (2.0*sigma_x - sigma_y - sigma_z)/3.0;
  Ds_x_x =  2.0/3.0;
  Ds_x_y = -1.0/3.0;
  Ds_x_z = -1.0/3.0;
  s_y = (2.0*sigma_y - sigma_x - sigma_z)/3.0;
  Ds_y_x = -1.0/3.0;
  Ds_y_y =  2.0/3.0;
  Ds_y_z = -1.0/3.0;
  s_z = (2.0*sigma_z - sigma_y - sigma_x)/3.0;
  Ds_z_x = -1.0/3.0;
  Ds_z_y = -1.0/3.0;
  Ds_z_z =  2.0/3.0;

  J_3 = s_x*s_y*s_z - s_x*pow(sigma_yz,2) - s_y*pow(sigma_zx,2) - s_z*pow(sigma_xy,2) + 2.0*sigma_xy*sigma_yz*sigma_zx;
  DJ_3_x = Ds_x_x*s_y*s_z + s_x*Ds_y_x*s_z + s_x*s_y*Ds_z_x
    - Ds_x_x*pow(sigma_yz,2)
    - Ds_y_x*pow(sigma_zx,2)
    - Ds_z_x*pow(sigma_xy,2);
  DJ_3_y = Ds_x_y*s_y*s_z + s_x*Ds_y_y*s_z + s_x*s_y*Ds_z_y  
    - Ds_x_y*pow(sigma_yz,2) 
    - Ds_y_y*pow(sigma_zx,2) 
    - Ds_z_y*pow(sigma_xy,2);
  DJ_3_z = Ds_x_z*s_y*s_z + s_x*Ds_y_z*s_z + s_x*s_y*Ds_z_z  
    - Ds_x_z*pow(sigma_yz,2) 
    - Ds_y_z*pow(sigma_zx,2) 
    - Ds_z_z*pow(sigma_xy,2);
  DJ_3_xy = - 2.0*s_z*sigma_xy + 2.0*sigma_yz*sigma_zx; 
  DJ_3_yz = - 2.0*s_x*sigma_yz + 2.0*sigma_xy*sigma_zx;
  DJ_3_zx = - 2.0*s_y*sigma_zx + 2.0*sigma_xy*sigma_yz;
    
  arg = -3.0*sqrt(6.0)*J_3/pow(t_star,3);
  arg_star = arg;
  if (arg >= 1.0-eps)
    {
      arg_star = 1.0-eps;
      arg = 1.0;
    }
  if (arg <= -1.0+eps)
    {
      arg_star = -1.0+eps;
      arg = -1.0;
    }
  Darg_J_3 = -3.0*sqrt(6.0)/pow(t_star,3);
  Darg_t = 9.0*sqrt(6.0)*J_3/pow(t_star,4);

  theta = asin(arg)/3.0;

  sin_theta = sin(theta);
  cos_theta = cos(theta);

  Dtheta_arg = 1.0/(3.0*sqrt(1.0-pow(arg_star,2)));
  Dtheta_x  = Dtheta_arg*(Darg_J_3*DJ_3_x + Darg_t*Dt_x);
  Dtheta_y  = Dtheta_arg*(Darg_J_3*DJ_3_y + Darg_t*Dt_y);
  Dtheta_z  = Dtheta_arg*(Darg_J_3*DJ_3_z + Darg_t*Dt_z);
  Dtheta_xy = Dtheta_arg*(Darg_J_3*DJ_3_xy + Darg_t*Dt_xy);
  Dtheta_yz = Dtheta_arg*(Darg_J_3*DJ_3_yz + Darg_t*Dt_yz);
  Dtheta_zx = Dtheta_arg*(Darg_J_3*DJ_3_zx + Darg_t*Dt_zx);

  //f
  f =  sigma_m*sin(phi) 
    + sigma_b*(cos_theta/sqrt(3.0) - (sin_theta*sin(phi))/3.0) 
    - c*cos(phi);
  
  Df_sigma_m = sin(phi);
  
  Df_sigma_b = (cos_theta/sqrt(3.0) - (sin_theta*sin(phi))/3.0);
  
  Df_theta = sigma_b*(-sin_theta/sqrt(3.0) - (cos_theta*sin(phi))/3.0);

  df[sXX] = Df_sigma_m*Dsigma_m_x + 
    Df_sigma_b*Dsigma_b_x + 
    Df_theta*Dtheta_x;
  df[sYY] = Df_sigma_m*Dsigma_m_y + 
    Df_sigma_b*Dsigma_b_y + 
    Df_theta*Dtheta_y;
  df[sZZ] = Df_sigma_m*Dsigma_m_z + 
    Df_sigma_b*Dsigma_b_z + 
    Df_theta*Dtheta_z;
  df[sXY] = Df_sigma_b*Dsigma_b_xy + 
    Df_theta*Dtheta_xy;
  df[sYZ] = Df_sigma_b*Dsigma_b_yz + 
    Df_theta*Dtheta_yz;
  df[sZX] = Df_sigma_b*Dsigma_b_zx + 
    Df_theta*Dtheta_zx;
  //r
  g =  sigma_m*sin(psi)
    + sigma_b*(cos_theta/sqrt(3.0) - (sin_theta*sin(psi))/3.0) 
    - c*cos(psi);
  
  Dg_sigma_m = sin(psi);
  
  Dg_sigma_b = (cos_theta/sqrt(3.0) - (sin_theta*sin(psi))/3.0);
  
  Dg_theta = sigma_b*(-sin_theta/sqrt(3.0) - (cos_theta*sin(psi))/3.0);
  
  r[sXX] = Dg_sigma_m*Dsigma_m_x + 
    Dg_sigma_b*Dsigma_b_x + 
    Dg_theta*Dtheta_x;
  r[sYY] = Dg_sigma_m*Dsigma_m_y + 
    Dg_sigma_b*Dsigma_b_y + 
    Dg_theta*Dtheta_y;
  r[sZZ] = Dg_sigma_m*Dsigma_m_z + 
    Dg_sigma_b*Dsigma_b_z + 
    Dg_theta*Dtheta_z;
  r[sXY] = Dg_sigma_b*Dsigma_b_xy + 
    Dg_theta*Dtheta_xy;
  r[sYZ] = Dg_sigma_b*Dsigma_b_yz + 
    Dg_theta*Dtheta_yz;
  r[sZX] = Dg_sigma_b*Dsigma_b_zx + 
    Dg_theta*Dtheta_zx;
}
