
inline void vonMises(const double& phi, const double& psi, const double& c, const double* stress, //inputs
                               double& f, double* df, double& g, double* r, double* dr) //outputs
{
  const int nSymTen=6,sXX=0,sYY=1,sZZ=2,sYZ=3,sZY=3,sXZ=4,sZX=4,sXY=5,sYX=5;
  double targ,dtarg[nSymTen],ddtarg[nSymTen*nSymTen],
    df_dtarg,
    dg_dtarg,ddg_dtarg;
  const double& stress_xx(stress[sXX]),
    stress_yy(stress[sYY]),
    stress_zz(stress[sZZ]),
    stress_yz(stress[sYZ]),
    stress_zx(stress[sZX]),
    stress_xy(stress[sXY]);

  targ = 1.0e-8 + pow(stress_xx - stress_yy, 2) + pow(stress_yy - stress_zz, 2) + pow(stress_zz - stress_xx, 2) + 6*pow(stress_xy, 2) + 6*pow(stress_yz, 2) + 6*pow(stress_zx, 2);

  dtarg[sXX] = -2*stress_yy - 2*stress_zz + 4*stress_xx;

  dtarg[sYY] = -2*stress_xx - 2*stress_zz + 4*stress_yy;

  dtarg[sZZ] = -2*stress_xx - 2*stress_yy + 4*stress_zz;

  dtarg[sYZ] = 12*stress_yz;

  dtarg[sZX] = 12*stress_zx;

  dtarg[sXY] = 12*stress_xy;

  ddtarg[sXX+ nSymTen*sXX] = 4;

  ddtarg[sXX+ nSymTen*sYY] = -2;

  ddtarg[sXX+ nSymTen*sZZ] = -2;

  ddtarg[sXX+ nSymTen*sYZ] = 0;

  ddtarg[sXX+ nSymTen*sZX] = 0;

  ddtarg[sXX+ nSymTen*sXY] = 0;

  ddtarg[sYY+ nSymTen*sXX] = -2;

  ddtarg[sYY+ nSymTen*sYY] = 4;

  ddtarg[sYY+ nSymTen*sZZ] = -2;

  ddtarg[sYY+ nSymTen*sYZ] = 0;

  ddtarg[sYY+ nSymTen*sZX] = 0;

  ddtarg[sYY+ nSymTen*sXY] = 0;

  ddtarg[sZZ+ nSymTen*sXX] = -2;

  ddtarg[sZZ+ nSymTen*sYY] = -2;

  ddtarg[sZZ+ nSymTen*sZZ] = 4;

  ddtarg[sZZ+ nSymTen*sYZ] = 0;

  ddtarg[sZZ+ nSymTen*sZX] = 0;

  ddtarg[sZZ+ nSymTen*sXY] = 0;

  ddtarg[sYZ+ nSymTen*sXX] = 0;

  ddtarg[sYZ+ nSymTen*sYY] = 0;

  ddtarg[sYZ+ nSymTen*sZZ] = 0;

  ddtarg[sYZ+ nSymTen*sYZ] = 12;

  ddtarg[sYZ+ nSymTen*sZX] = 0;

  ddtarg[sYZ+ nSymTen*sXY] = 0;

  ddtarg[sZX+ nSymTen*sXX] = 0;

  ddtarg[sZX+ nSymTen*sYY] = 0;

  ddtarg[sZX+ nSymTen*sZZ] = 0;

  ddtarg[sZX+ nSymTen*sYZ] = 0;

  ddtarg[sZX+ nSymTen*sZX] = 12;

  ddtarg[sZX+ nSymTen*sXY] = 0;

  ddtarg[sXY+ nSymTen*sXX] = 0;

  ddtarg[sXY+ nSymTen*sYY] = 0;

  ddtarg[sXY+ nSymTen*sZZ] = 0;

  ddtarg[sXY+ nSymTen*sYZ] = 0;

  ddtarg[sXY+ nSymTen*sZX] = 0;

  ddtarg[sXY+ nSymTen*sXY] = 12;

  f = -2*c + sqrt(2)*sqrt(targ)/2;

  df_dtarg = sqrt(2)/(4*sqrt(targ));

  df[sXX] = df_dtarg*dtarg[sXX];

  df[sYY] = df_dtarg*dtarg[sYY];

  df[sZZ] = df_dtarg*dtarg[sZZ];

  df[sYZ] = df_dtarg*dtarg[sYZ];

  df[sZX] = df_dtarg*dtarg[sZX];

  df[sXY] = df_dtarg*dtarg[sXY];

  g = -2*c + sqrt(2)*sqrt(targ)/2;

  dg_dtarg = sqrt(2)/(4*sqrt(targ));

  r[sXX] = dg_dtarg*dtarg[sXX];

  r[sYY] = dg_dtarg*dtarg[sYY];

  r[sZZ] = dg_dtarg*dtarg[sZZ];

  r[sYZ] = dg_dtarg*dtarg[sYZ];

  r[sZX] = dg_dtarg*dtarg[sZX];

  r[sXY] = dg_dtarg*dtarg[sXY];

  ddg_dtarg = -sqrt(2)/(8*pow(targ, 3.0/2.0));

  dr[sXX+ nSymTen*sXX] = dg_dtarg*ddtarg[sXX+nSymTen*sXX]  + ddg_dtarg*dtarg[sXX]*dtarg[sXX];

  dr[sXX+ nSymTen*sYY] = dg_dtarg*ddtarg[sXX+nSymTen*sYY]  + ddg_dtarg*dtarg[sYY]*dtarg[sXX];

  dr[sXX+ nSymTen*sZZ] = dg_dtarg*ddtarg[sXX+nSymTen*sZZ]  + ddg_dtarg*dtarg[sZZ]*dtarg[sXX];

  dr[sXX+ nSymTen*sYZ] = dg_dtarg*ddtarg[sXX+nSymTen*sYZ]  + ddg_dtarg*dtarg[sYZ]*dtarg[sXX];

  dr[sXX+ nSymTen*sZX] = dg_dtarg*ddtarg[sXX+nSymTen*sZX]  + ddg_dtarg*dtarg[sZX]*dtarg[sXX];

  dr[sXX+ nSymTen*sXY] = dg_dtarg*ddtarg[sXX+nSymTen*sXY]  + ddg_dtarg*dtarg[sXY]*dtarg[sXX];

  dr[sYY+ nSymTen*sXX] = dg_dtarg*ddtarg[sYY+nSymTen*sXX]  + ddg_dtarg*dtarg[sXX]*dtarg[sYY];

  dr[sYY+ nSymTen*sYY] = dg_dtarg*ddtarg[sYY+nSymTen*sYY]  + ddg_dtarg*dtarg[sYY]*dtarg[sYY];

  dr[sYY+ nSymTen*sZZ] = dg_dtarg*ddtarg[sYY+nSymTen*sZZ]  + ddg_dtarg*dtarg[sZZ]*dtarg[sYY];

  dr[sYY+ nSymTen*sYZ] = dg_dtarg*ddtarg[sYY+nSymTen*sYZ]  + ddg_dtarg*dtarg[sYZ]*dtarg[sYY];

  dr[sYY+ nSymTen*sZX] = dg_dtarg*ddtarg[sYY+nSymTen*sZX]  + ddg_dtarg*dtarg[sZX]*dtarg[sYY];

  dr[sYY+ nSymTen*sXY] = dg_dtarg*ddtarg[sYY+nSymTen*sXY]  + ddg_dtarg*dtarg[sXY]*dtarg[sYY];

  dr[sZZ+ nSymTen*sXX] = dg_dtarg*ddtarg[sZZ+nSymTen*sXX]  + ddg_dtarg*dtarg[sXX]*dtarg[sZZ];

  dr[sZZ+ nSymTen*sYY] = dg_dtarg*ddtarg[sZZ+nSymTen*sYY]  + ddg_dtarg*dtarg[sYY]*dtarg[sZZ];

  dr[sZZ+ nSymTen*sZZ] = dg_dtarg*ddtarg[sZZ+nSymTen*sZZ]  + ddg_dtarg*dtarg[sZZ]*dtarg[sZZ];

  dr[sZZ+ nSymTen*sYZ] = dg_dtarg*ddtarg[sZZ+nSymTen*sYZ]  + ddg_dtarg*dtarg[sYZ]*dtarg[sZZ];

  dr[sZZ+ nSymTen*sZX] = dg_dtarg*ddtarg[sZZ+nSymTen*sZX]  + ddg_dtarg*dtarg[sZX]*dtarg[sZZ];

  dr[sZZ+ nSymTen*sXY] = dg_dtarg*ddtarg[sZZ+nSymTen*sXY]  + ddg_dtarg*dtarg[sXY]*dtarg[sZZ];

  dr[sYZ+ nSymTen*sXX] = dg_dtarg*ddtarg[sYZ+nSymTen*sXX]  + ddg_dtarg*dtarg[sXX]*dtarg[sYZ];

  dr[sYZ+ nSymTen*sYY] = dg_dtarg*ddtarg[sYZ+nSymTen*sYY]  + ddg_dtarg*dtarg[sYY]*dtarg[sYZ];

  dr[sYZ+ nSymTen*sZZ] = dg_dtarg*ddtarg[sYZ+nSymTen*sZZ]  + ddg_dtarg*dtarg[sZZ]*dtarg[sYZ];

  dr[sYZ+ nSymTen*sYZ] = dg_dtarg*ddtarg[sYZ+nSymTen*sYZ]  + ddg_dtarg*dtarg[sYZ]*dtarg[sYZ];

  dr[sYZ+ nSymTen*sZX] = dg_dtarg*ddtarg[sYZ+nSymTen*sZX]  + ddg_dtarg*dtarg[sZX]*dtarg[sYZ];

  dr[sYZ+ nSymTen*sXY] = dg_dtarg*ddtarg[sYZ+nSymTen*sXY]  + ddg_dtarg*dtarg[sXY]*dtarg[sYZ];

  dr[sZX+ nSymTen*sXX] = dg_dtarg*ddtarg[sZX+nSymTen*sXX]  + ddg_dtarg*dtarg[sXX]*dtarg[sZX];

  dr[sZX+ nSymTen*sYY] = dg_dtarg*ddtarg[sZX+nSymTen*sYY]  + ddg_dtarg*dtarg[sYY]*dtarg[sZX];

  dr[sZX+ nSymTen*sZZ] = dg_dtarg*ddtarg[sZX+nSymTen*sZZ]  + ddg_dtarg*dtarg[sZZ]*dtarg[sZX];

  dr[sZX+ nSymTen*sYZ] = dg_dtarg*ddtarg[sZX+nSymTen*sYZ]  + ddg_dtarg*dtarg[sYZ]*dtarg[sZX];

  dr[sZX+ nSymTen*sZX] = dg_dtarg*ddtarg[sZX+nSymTen*sZX]  + ddg_dtarg*dtarg[sZX]*dtarg[sZX];

  dr[sZX+ nSymTen*sXY] = dg_dtarg*ddtarg[sZX+nSymTen*sXY]  + ddg_dtarg*dtarg[sXY]*dtarg[sZX];

  dr[sXY+ nSymTen*sXX] = dg_dtarg*ddtarg[sXY+nSymTen*sXX]  + ddg_dtarg*dtarg[sXX]*dtarg[sXY];

  dr[sXY+ nSymTen*sYY] = dg_dtarg*ddtarg[sXY+nSymTen*sYY]  + ddg_dtarg*dtarg[sYY]*dtarg[sXY];

  dr[sXY+ nSymTen*sZZ] = dg_dtarg*ddtarg[sXY+nSymTen*sZZ]  + ddg_dtarg*dtarg[sZZ]*dtarg[sXY];

  dr[sXY+ nSymTen*sYZ] = dg_dtarg*ddtarg[sXY+nSymTen*sYZ]  + ddg_dtarg*dtarg[sYZ]*dtarg[sXY];

  dr[sXY+ nSymTen*sZX] = dg_dtarg*ddtarg[sXY+nSymTen*sZX]  + ddg_dtarg*dtarg[sZX]*dtarg[sXY];

  dr[sXY+ nSymTen*sXY] = dg_dtarg*ddtarg[sXY+nSymTen*sXY]  + ddg_dtarg*dtarg[sXY]*dtarg[sXY];

}
