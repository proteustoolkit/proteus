
inline void mohrCoulomb(const double& phi, const double& psi, const double& c, const double* stress, //inputs
                               double& f, double* df, double& g, double* r, double* dr) //outputs
{
  const int nSymTen=6,sXX=0,sYY=1,sZZ=2,sYZ=3,sZY=3,sXZ=4,sZX=4,sXY=5,sYX=5;
  const int SKEY=0,TARGKEY=1,J3KEY=2;
  double s,ds[nSymTen],
    targ,dtarg[nSymTen],ddtarg[nSymTen*nSymTen],
    J3,dJ3[nSymTen],ddJ3[nSymTen*nSymTen],
    df_dtmp[3],
    dg_dtmp[3],ddg_dtmp[3*3],theta;
  const double& stress_xx(stress[sXX]),
    stress_yy(stress[sYY]),
    stress_zz(stress[sZZ]),
    stress_yz(stress[sYZ]),
    stress_zx(stress[sZX]),
    stress_xy(stress[sXY]);

  s = sqrt(3)*(stress_xx + stress_yy + stress_zz)/3;

  ds[sXX] = sqrt(3)/3;

  ds[sYY] = sqrt(3)/3;

  ds[sZZ] = sqrt(3)/3;

  ds[sYZ] = 0;

  ds[sZX] = 0;

  ds[sXY] = 0;

  targ = pow(stress_xx - stress_yy, 2) + pow(stress_yy - stress_zz, 2) + pow(stress_zz - stress_xx, 2) + 6*pow(stress_xy, 2) + 6*pow(stress_yz, 2) + 6*pow(stress_zx, 2);

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

  if (targ < 1.0e-8)
    {
      targ = 1.0e-8;
    }

  J3 = -pow(stress_xy, 2)*(-stress_xx/3 - stress_yy/3 + 2*stress_zz/3) - pow(stress_yz, 2)*(-stress_yy/3 - stress_zz/3 + 2*stress_xx/3) - pow(stress_zx, 2)*(-stress_xx/3 - stress_zz/3 + 2*stress_yy/3) + (-stress_xx/3 - stress_zz/3 + 2*stress_yy/3)*(-stress_yy/3 - stress_zz/3 + 2*stress_xx/3)*(-stress_xx/3 - stress_yy/3 + 2*stress_zz/3) + 2*stress_xy*stress_yz*stress_zx;

  dJ3[sXX] = -(-stress_xx/3 - stress_zz/3 + 2*stress_yy/3)*(-stress_yy/3 - stress_zz/3 + 2*stress_xx/3)/3 - (-stress_yy/3 - stress_zz/3 + 2*stress_xx/3)*(-stress_xx/3 - stress_yy/3 + 2*stress_zz/3)/3 + 2*(-stress_xx/3 - stress_zz/3 + 2*stress_yy/3)*(-stress_xx/3 - stress_yy/3 + 2*stress_zz/3)/3 - 2*pow(stress_yz, 2)/3 + pow(stress_xy, 2)/3 + pow(stress_zx, 2)/3;

  dJ3[sYY] = -(-stress_xx/3 - stress_zz/3 + 2*stress_yy/3)*(-stress_yy/3 - stress_zz/3 + 2*stress_xx/3)/3 - (-stress_xx/3 - stress_zz/3 + 2*stress_yy/3)*(-stress_xx/3 - stress_yy/3 + 2*stress_zz/3)/3 + 2*(-stress_yy/3 - stress_zz/3 + 2*stress_xx/3)*(-stress_xx/3 - stress_yy/3 + 2*stress_zz/3)/3 - 2*pow(stress_zx, 2)/3 + pow(stress_xy, 2)/3 + pow(stress_yz, 2)/3;

  dJ3[sZZ] = -(-stress_xx/3 - stress_zz/3 + 2*stress_yy/3)*(-stress_xx/3 - stress_yy/3 + 2*stress_zz/3)/3 - (-stress_yy/3 - stress_zz/3 + 2*stress_xx/3)*(-stress_xx/3 - stress_yy/3 + 2*stress_zz/3)/3 + 2*(-stress_xx/3 - stress_zz/3 + 2*stress_yy/3)*(-stress_yy/3 - stress_zz/3 + 2*stress_xx/3)/3 - 2*pow(stress_xy, 2)/3 + pow(stress_yz, 2)/3 + pow(stress_zx, 2)/3;

  dJ3[sYZ] = -2*stress_yz*(-stress_yy/3 - stress_zz/3 + 2*stress_xx/3) + 2*stress_xy*stress_zx;

  dJ3[sZX] = -2*stress_zx*(-stress_xx/3 - stress_zz/3 + 2*stress_yy/3) + 2*stress_xy*stress_yz;

  dJ3[sXY] = -2*stress_xy*(-stress_xx/3 - stress_yy/3 + 2*stress_zz/3) + 2*stress_yz*stress_zx;

  ddJ3[sXX+ nSymTen*sXX] = -2*stress_yy/9 - 2*stress_zz/9 + 4*stress_xx/9;

  ddJ3[sXX+ nSymTen*sYY] = -2*stress_xx/9 - 2*stress_yy/9 + 4*stress_zz/9;

  ddJ3[sXX+ nSymTen*sZZ] = -2*stress_xx/9 - 2*stress_zz/9 + 4*stress_yy/9;

  ddJ3[sXX+ nSymTen*sYZ] = -4*stress_yz/3;

  ddJ3[sXX+ nSymTen*sZX] = 2*stress_zx/3;

  ddJ3[sXX+ nSymTen*sXY] = 2*stress_xy/3;

  ddJ3[sYY+ nSymTen*sXX] = -2*stress_xx/9 - 2*stress_yy/9 + 4*stress_zz/9;

  ddJ3[sYY+ nSymTen*sYY] = -2*stress_xx/9 - 2*stress_zz/9 + 4*stress_yy/9;

  ddJ3[sYY+ nSymTen*sZZ] = -2*stress_yy/9 - 2*stress_zz/9 + 4*stress_xx/9;

  ddJ3[sYY+ nSymTen*sYZ] = 2*stress_yz/3;

  ddJ3[sYY+ nSymTen*sZX] = -4*stress_zx/3;

  ddJ3[sYY+ nSymTen*sXY] = 2*stress_xy/3;

  ddJ3[sZZ+ nSymTen*sXX] = -2*stress_xx/9 - 2*stress_zz/9 + 4*stress_yy/9;

  ddJ3[sZZ+ nSymTen*sYY] = -2*stress_yy/9 - 2*stress_zz/9 + 4*stress_xx/9;

  ddJ3[sZZ+ nSymTen*sZZ] = -2*stress_xx/9 - 2*stress_yy/9 + 4*stress_zz/9;

  ddJ3[sZZ+ nSymTen*sYZ] = 2*stress_yz/3;

  ddJ3[sZZ+ nSymTen*sZX] = 2*stress_zx/3;

  ddJ3[sZZ+ nSymTen*sXY] = -4*stress_xy/3;

  ddJ3[sYZ+ nSymTen*sXX] = -4*stress_yz/3;

  ddJ3[sYZ+ nSymTen*sYY] = 2*stress_yz/3;

  ddJ3[sYZ+ nSymTen*sZZ] = 2*stress_yz/3;

  ddJ3[sYZ+ nSymTen*sYZ] = -4*stress_xx/3 + 2*stress_yy/3 + 2*stress_zz/3;

  ddJ3[sYZ+ nSymTen*sZX] = 2*stress_xy;

  ddJ3[sYZ+ nSymTen*sXY] = 2*stress_zx;

  ddJ3[sZX+ nSymTen*sXX] = 2*stress_zx/3;

  ddJ3[sZX+ nSymTen*sYY] = -4*stress_zx/3;

  ddJ3[sZX+ nSymTen*sZZ] = 2*stress_zx/3;

  ddJ3[sZX+ nSymTen*sYZ] = 2*stress_xy;

  ddJ3[sZX+ nSymTen*sZX] = -4*stress_yy/3 + 2*stress_xx/3 + 2*stress_zz/3;

  ddJ3[sZX+ nSymTen*sXY] = 2*stress_yz;

  ddJ3[sXY+ nSymTen*sXX] = 2*stress_xy/3;

  ddJ3[sXY+ nSymTen*sYY] = 2*stress_xy/3;

  ddJ3[sXY+ nSymTen*sZZ] = -4*stress_xy/3;

  ddJ3[sXY+ nSymTen*sYZ] = 2*stress_zx;

  ddJ3[sXY+ nSymTen*sZX] = 2*stress_yz;

  ddJ3[sXY+ nSymTen*sXY] = -4*stress_zz/3 + 2*stress_xx/3 + 2*stress_yy/3;

  double tmpval = 27*J3*sqrt(2)/pow(targ, 3.0/2.0);
  //make sure we're exactly in the domain of asin
  if (tmpval > 1.0)
    tmpval = 1.0;
  if (tmpval < -1.0)
    tmpval = -1.0;

  theta = -asin(tmpval)/3;

  f = -c*cos(phi) + sqrt(2)*sqrt(targ)*(sqrt(3)*cos(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)/3 + sin(phi)*sin(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)/3)/2 + s*sqrt(3)*sin(phi)/3;

  df_dtmp[SKEY] = sqrt(3)*sin(phi)/3;

  df_dtmp[TARGKEY] = sqrt(2)*sqrt(targ)*(9*J3*sqrt(6)*sin(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)/(2*pow(targ, 5.0/2.0)*sqrt(1 - 1458*pow(J3, 2)/pow(targ, 3))) - 9*J3*sqrt(2)*cos(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)*sin(phi)/(2*pow(targ, 5.0/2.0)*sqrt(1 - 1458*pow(J3, 2)/pow(targ, 3))))/2 + sqrt(2)*(sqrt(3)*cos(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)/3 + sin(phi)*sin(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)/3)/(4*sqrt(targ));

  df_dtmp[J3KEY] = sqrt(2)*sqrt(targ)*(-3*sqrt(6)*sin(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)/(pow(targ, 3.0/2.0)*sqrt(1 - 1458*pow(J3, 2)/pow(targ, 3))) + 3*sqrt(2)*cos(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)*sin(phi)/(pow(targ, 3.0/2.0)*sqrt(1 - 1458*pow(J3, 2)/pow(targ, 3))))/2;

  df[sXX] = df_dtmp[SKEY]*ds[sXX] + df_dtmp[TARGKEY]*dtarg[sXX] + df_dtmp[J3KEY]*dJ3[sXX];

  df[sYY] = df_dtmp[SKEY]*ds[sYY] + df_dtmp[TARGKEY]*dtarg[sYY] + df_dtmp[J3KEY]*dJ3[sYY];

  df[sZZ] = df_dtmp[SKEY]*ds[sZZ] + df_dtmp[TARGKEY]*dtarg[sZZ] + df_dtmp[J3KEY]*dJ3[sZZ];

  df[sYZ] = df_dtmp[SKEY]*ds[sYZ] + df_dtmp[TARGKEY]*dtarg[sYZ] + df_dtmp[J3KEY]*dJ3[sYZ];

  df[sZX] = df_dtmp[SKEY]*ds[sZX] + df_dtmp[TARGKEY]*dtarg[sZX] + df_dtmp[J3KEY]*dJ3[sZX];

  df[sXY] = df_dtmp[SKEY]*ds[sXY] + df_dtmp[TARGKEY]*dtarg[sXY] + df_dtmp[J3KEY]*dJ3[sXY];

  g = -c*cos(psi) + sqrt(2)*sqrt(targ)*(sqrt(3)*cos(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)/3 + sin(psi)*sin(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)/3)/2 + s*sqrt(3)*sin(psi)/3;

  dg_dtmp[SKEY] = sqrt(3)*sin(psi)/3;

  dg_dtmp[TARGKEY] = sqrt(2)*sqrt(targ)*(9*J3*sqrt(6)*sin(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)/(2*pow(targ, 5.0/2.0)*sqrt(1 - 1458*pow(J3, 2)/pow(targ, 3))) - 9*J3*sqrt(2)*cos(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)*sin(psi)/(2*pow(targ, 5.0/2.0)*sqrt(1 - 1458*pow(J3, 2)/pow(targ, 3))))/2 + sqrt(2)*(sqrt(3)*cos(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)/3 + sin(psi)*sin(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)/3)/(4*sqrt(targ));

  dg_dtmp[J3KEY] = sqrt(2)*sqrt(targ)*(-3*sqrt(6)*sin(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)/(pow(targ, 3.0/2.0)*sqrt(1 - 1458*pow(J3, 2)/pow(targ, 3))) + 3*sqrt(2)*cos(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)*sin(psi)/(pow(targ, 3.0/2.0)*sqrt(1 - 1458*pow(J3, 2)/pow(targ, 3))))/2;

  r[sXX] = dg_dtmp[SKEY]*ds[sXX] + dg_dtmp[TARGKEY]*dtarg[sXX] + dg_dtmp[J3KEY]*dJ3[sXX];

  r[sYY] = dg_dtmp[SKEY]*ds[sYY] + dg_dtmp[TARGKEY]*dtarg[sYY] + dg_dtmp[J3KEY]*dJ3[sYY];

  r[sZZ] = dg_dtmp[SKEY]*ds[sZZ] + dg_dtmp[TARGKEY]*dtarg[sZZ] + dg_dtmp[J3KEY]*dJ3[sZZ];

  r[sYZ] = dg_dtmp[SKEY]*ds[sYZ] + dg_dtmp[TARGKEY]*dtarg[sYZ] + dg_dtmp[J3KEY]*dJ3[sYZ];

  r[sZX] = dg_dtmp[SKEY]*ds[sZX] + dg_dtmp[TARGKEY]*dtarg[sZX] + dg_dtmp[J3KEY]*dJ3[sZX];

  r[sXY] = dg_dtmp[SKEY]*ds[sXY] + dg_dtmp[TARGKEY]*dtarg[sXY] + dg_dtmp[J3KEY]*dJ3[sXY];

  ddg_dtmp[SKEY+3*SKEY] = 0;

  ddg_dtmp[SKEY+3*TARGKEY] = 0;

  ddg_dtmp[SKEY+3*J3KEY] = 0;

  ddg_dtmp[TARGKEY+3*SKEY] = 0;

  ddg_dtmp[TARGKEY+3*TARGKEY] = sqrt(2)*sqrt(targ)*(-19683*sqrt(6)*pow(J3, 3)*sin(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)/(2*pow(targ, 13.0/2.0)*pow(1 - 1458*pow(J3, 2)/pow(targ, 3), 3.0/2.0)) - 243*sqrt(3)*pow(J3, 2)*cos(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)/(2*pow(targ, 5)*(1 - 1458*pow(J3, 2)/pow(targ, 3))) - 243*pow(J3, 2)*sin(psi)*sin(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)/(2*pow(targ, 5)*(1 - 1458*pow(J3, 2)/pow(targ, 3))) - 45*J3*sqrt(6)*sin(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)/(4*pow(targ, 7.0/2.0)*sqrt(1 - 1458*pow(J3, 2)/pow(targ, 3))) + 45*J3*sqrt(2)*cos(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)*sin(psi)/(4*pow(targ, 7.0/2.0)*sqrt(1 - 1458*pow(J3, 2)/pow(targ, 3))) + 19683*sqrt(2)*pow(J3, 3)*cos(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)*sin(psi)/(2*pow(targ, 13.0/2.0)*pow(1 - 1458*pow(J3, 2)/pow(targ, 3), 3.0/2.0)))/2 + sqrt(2)*(9*J3*sqrt(6)*sin(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)/(2*pow(targ, 5.0/2.0)*sqrt(1 - 1458*pow(J3, 2)/pow(targ, 3))) - 9*J3*sqrt(2)*cos(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)*sin(psi)/(2*pow(targ, 5.0/2.0)*sqrt(1 - 1458*pow(J3, 2)/pow(targ, 3))))/(2*sqrt(targ)) - sqrt(2)*(sqrt(3)*cos(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)/3 + sin(psi)*sin(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)/3)/(8*pow(targ, 3.0/2.0));

  ddg_dtmp[TARGKEY+3*J3KEY] = sqrt(2)*sqrt(targ)*(9*sqrt(6)*sin(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)/(2*pow(targ, 5.0/2.0)*sqrt(1 - 1458*pow(J3, 2)/pow(targ, 3))) + 81*J3*sqrt(3)*cos(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)/(pow(targ, 4)*(1 - 1458*pow(J3, 2)/pow(targ, 3))) + 81*J3*sin(psi)*sin(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)/(pow(targ, 4)*(1 - 1458*pow(J3, 2)/pow(targ, 3))) + 6561*sqrt(6)*pow(J3, 2)*sin(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)/(pow(targ, 11.0/2.0)*pow(1 - 1458*pow(J3, 2)/pow(targ, 3), 3.0/2.0)) - 9*sqrt(2)*cos(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)*sin(psi)/(2*pow(targ, 5.0/2.0)*sqrt(1 - 1458*pow(J3, 2)/pow(targ, 3))) - 6561*sqrt(2)*pow(J3, 2)*cos(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)*sin(psi)/(pow(targ, 11.0/2.0)*pow(1 - 1458*pow(J3, 2)/pow(targ, 3), 3.0/2.0)))/2 + sqrt(2)*(-3*sqrt(6)*sin(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)/(pow(targ, 3.0/2.0)*sqrt(1 - 1458*pow(J3, 2)/pow(targ, 3))) + 3*sqrt(2)*cos(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)*sin(psi)/(pow(targ, 3.0/2.0)*sqrt(1 - 1458*pow(J3, 2)/pow(targ, 3))))/(4*sqrt(targ));

  ddg_dtmp[J3KEY+3*SKEY] = 0;

  ddg_dtmp[J3KEY+3*TARGKEY] = sqrt(2)*sqrt(targ)*(9*sqrt(6)*sin(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)/(2*pow(targ, 5.0/2.0)*sqrt(1 - 1458*pow(J3, 2)/pow(targ, 3))) + 81*J3*sqrt(3)*cos(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)/(pow(targ, 4)*(1 - 1458*pow(J3, 2)/pow(targ, 3))) + 81*J3*sin(psi)*sin(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)/(pow(targ, 4)*(1 - 1458*pow(J3, 2)/pow(targ, 3))) + 6561*sqrt(6)*pow(J3, 2)*sin(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)/(pow(targ, 11.0/2.0)*pow(1 - 1458*pow(J3, 2)/pow(targ, 3), 3.0/2.0)) - 9*sqrt(2)*cos(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)*sin(psi)/(2*pow(targ, 5.0/2.0)*sqrt(1 - 1458*pow(J3, 2)/pow(targ, 3))) - 6561*sqrt(2)*pow(J3, 2)*cos(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)*sin(psi)/(pow(targ, 11.0/2.0)*pow(1 - 1458*pow(J3, 2)/pow(targ, 3), 3.0/2.0)))/2 + sqrt(2)*(-3*sqrt(6)*sin(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)/(pow(targ, 3.0/2.0)*sqrt(1 - 1458*pow(J3, 2)/pow(targ, 3))) + 3*sqrt(2)*cos(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)*sin(psi)/(pow(targ, 3.0/2.0)*sqrt(1 - 1458*pow(J3, 2)/pow(targ, 3))))/(4*sqrt(targ));

  ddg_dtmp[J3KEY+3*J3KEY] = sqrt(2)*sqrt(targ)*(-54*sqrt(3)*cos(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)/(pow(targ, 3)*(1 - 1458*pow(J3, 2)/pow(targ, 3))) - 54*sin(psi)*sin(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)/(pow(targ, 3)*(1 - 1458*pow(J3, 2)/pow(targ, 3))) - 4374*J3*sqrt(6)*sin(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)/(pow(targ, 9.0/2.0)*pow(1 - 1458*pow(J3, 2)/pow(targ, 3), 3.0/2.0)) + 4374*J3*sqrt(2)*cos(asin(27*J3*sqrt(2)/pow(targ, 3.0/2.0))/3)*sin(psi)/(pow(targ, 9.0/2.0)*pow(1 - 1458*pow(J3, 2)/pow(targ, 3), 3.0/2.0)))/2;

  dr[sXX+ nSymTen*sXX] = dg_dtmp[TARGKEY]*ddtarg[sXX+nSymTen*sXX] + dg_dtmp[J3KEY]*ddJ3[sXX+nSymTen*sXX]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sXX]*ds[sXX]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sXX]*ds[sXX]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sXX]*ds[sXX]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sXX]*dtarg[sXX]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sXX]*dtarg[sXX]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sXX]*dtarg[sXX]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sXX]*dJ3[sXX]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sXX]*dJ3[sXX]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sXX]*dJ3[sXX];

  dr[sXX+ nSymTen*sYY] = dg_dtmp[TARGKEY]*ddtarg[sXX+nSymTen*sYY] + dg_dtmp[J3KEY]*ddJ3[sXX+nSymTen*sYY]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sYY]*ds[sXX]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sYY]*ds[sXX]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sYY]*ds[sXX]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sYY]*dtarg[sXX]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sYY]*dtarg[sXX]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sYY]*dtarg[sXX]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sYY]*dJ3[sXX]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sYY]*dJ3[sXX]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sYY]*dJ3[sXX];

  dr[sXX+ nSymTen*sZZ] = dg_dtmp[TARGKEY]*ddtarg[sXX+nSymTen*sZZ] + dg_dtmp[J3KEY]*ddJ3[sXX+nSymTen*sZZ]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sZZ]*ds[sXX]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sZZ]*ds[sXX]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sZZ]*ds[sXX]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sZZ]*dtarg[sXX]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sZZ]*dtarg[sXX]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sZZ]*dtarg[sXX]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sZZ]*dJ3[sXX]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sZZ]*dJ3[sXX]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sZZ]*dJ3[sXX];

  dr[sXX+ nSymTen*sYZ] = dg_dtmp[TARGKEY]*ddtarg[sXX+nSymTen*sYZ] + dg_dtmp[J3KEY]*ddJ3[sXX+nSymTen*sYZ]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sYZ]*ds[sXX]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sYZ]*ds[sXX]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sYZ]*ds[sXX]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sYZ]*dtarg[sXX]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sYZ]*dtarg[sXX]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sYZ]*dtarg[sXX]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sYZ]*dJ3[sXX]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sYZ]*dJ3[sXX]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sYZ]*dJ3[sXX];

  dr[sXX+ nSymTen*sZX] = dg_dtmp[TARGKEY]*ddtarg[sXX+nSymTen*sZX] + dg_dtmp[J3KEY]*ddJ3[sXX+nSymTen*sZX]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sZX]*ds[sXX]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sZX]*ds[sXX]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sZX]*ds[sXX]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sZX]*dtarg[sXX]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sZX]*dtarg[sXX]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sZX]*dtarg[sXX]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sZX]*dJ3[sXX]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sZX]*dJ3[sXX]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sZX]*dJ3[sXX];

  dr[sXX+ nSymTen*sXY] = dg_dtmp[TARGKEY]*ddtarg[sXX+nSymTen*sXY] + dg_dtmp[J3KEY]*ddJ3[sXX+nSymTen*sXY]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sXY]*ds[sXX]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sXY]*ds[sXX]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sXY]*ds[sXX]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sXY]*dtarg[sXX]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sXY]*dtarg[sXX]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sXY]*dtarg[sXX]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sXY]*dJ3[sXX]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sXY]*dJ3[sXX]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sXY]*dJ3[sXX];

  dr[sYY+ nSymTen*sXX] = dg_dtmp[TARGKEY]*ddtarg[sYY+nSymTen*sXX] + dg_dtmp[J3KEY]*ddJ3[sYY+nSymTen*sXX]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sXX]*ds[sYY]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sXX]*ds[sYY]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sXX]*ds[sYY]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sXX]*dtarg[sYY]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sXX]*dtarg[sYY]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sXX]*dtarg[sYY]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sXX]*dJ3[sYY]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sXX]*dJ3[sYY]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sXX]*dJ3[sYY];

  dr[sYY+ nSymTen*sYY] = dg_dtmp[TARGKEY]*ddtarg[sYY+nSymTen*sYY] + dg_dtmp[J3KEY]*ddJ3[sYY+nSymTen*sYY]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sYY]*ds[sYY]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sYY]*ds[sYY]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sYY]*ds[sYY]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sYY]*dtarg[sYY]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sYY]*dtarg[sYY]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sYY]*dtarg[sYY]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sYY]*dJ3[sYY]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sYY]*dJ3[sYY]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sYY]*dJ3[sYY];

  dr[sYY+ nSymTen*sZZ] = dg_dtmp[TARGKEY]*ddtarg[sYY+nSymTen*sZZ] + dg_dtmp[J3KEY]*ddJ3[sYY+nSymTen*sZZ]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sZZ]*ds[sYY]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sZZ]*ds[sYY]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sZZ]*ds[sYY]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sZZ]*dtarg[sYY]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sZZ]*dtarg[sYY]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sZZ]*dtarg[sYY]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sZZ]*dJ3[sYY]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sZZ]*dJ3[sYY]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sZZ]*dJ3[sYY];

  dr[sYY+ nSymTen*sYZ] = dg_dtmp[TARGKEY]*ddtarg[sYY+nSymTen*sYZ] + dg_dtmp[J3KEY]*ddJ3[sYY+nSymTen*sYZ]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sYZ]*ds[sYY]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sYZ]*ds[sYY]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sYZ]*ds[sYY]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sYZ]*dtarg[sYY]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sYZ]*dtarg[sYY]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sYZ]*dtarg[sYY]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sYZ]*dJ3[sYY]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sYZ]*dJ3[sYY]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sYZ]*dJ3[sYY];

  dr[sYY+ nSymTen*sZX] = dg_dtmp[TARGKEY]*ddtarg[sYY+nSymTen*sZX] + dg_dtmp[J3KEY]*ddJ3[sYY+nSymTen*sZX]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sZX]*ds[sYY]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sZX]*ds[sYY]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sZX]*ds[sYY]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sZX]*dtarg[sYY]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sZX]*dtarg[sYY]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sZX]*dtarg[sYY]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sZX]*dJ3[sYY]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sZX]*dJ3[sYY]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sZX]*dJ3[sYY];

  dr[sYY+ nSymTen*sXY] = dg_dtmp[TARGKEY]*ddtarg[sYY+nSymTen*sXY] + dg_dtmp[J3KEY]*ddJ3[sYY+nSymTen*sXY]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sXY]*ds[sYY]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sXY]*ds[sYY]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sXY]*ds[sYY]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sXY]*dtarg[sYY]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sXY]*dtarg[sYY]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sXY]*dtarg[sYY]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sXY]*dJ3[sYY]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sXY]*dJ3[sYY]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sXY]*dJ3[sYY];

  dr[sZZ+ nSymTen*sXX] = dg_dtmp[TARGKEY]*ddtarg[sZZ+nSymTen*sXX] + dg_dtmp[J3KEY]*ddJ3[sZZ+nSymTen*sXX]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sXX]*ds[sZZ]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sXX]*ds[sZZ]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sXX]*ds[sZZ]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sXX]*dtarg[sZZ]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sXX]*dtarg[sZZ]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sXX]*dtarg[sZZ]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sXX]*dJ3[sZZ]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sXX]*dJ3[sZZ]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sXX]*dJ3[sZZ];

  dr[sZZ+ nSymTen*sYY] = dg_dtmp[TARGKEY]*ddtarg[sZZ+nSymTen*sYY] + dg_dtmp[J3KEY]*ddJ3[sZZ+nSymTen*sYY]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sYY]*ds[sZZ]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sYY]*ds[sZZ]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sYY]*ds[sZZ]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sYY]*dtarg[sZZ]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sYY]*dtarg[sZZ]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sYY]*dtarg[sZZ]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sYY]*dJ3[sZZ]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sYY]*dJ3[sZZ]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sYY]*dJ3[sZZ];

  dr[sZZ+ nSymTen*sZZ] = dg_dtmp[TARGKEY]*ddtarg[sZZ+nSymTen*sZZ] + dg_dtmp[J3KEY]*ddJ3[sZZ+nSymTen*sZZ]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sZZ]*ds[sZZ]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sZZ]*ds[sZZ]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sZZ]*ds[sZZ]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sZZ]*dtarg[sZZ]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sZZ]*dtarg[sZZ]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sZZ]*dtarg[sZZ]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sZZ]*dJ3[sZZ]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sZZ]*dJ3[sZZ]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sZZ]*dJ3[sZZ];

  dr[sZZ+ nSymTen*sYZ] = dg_dtmp[TARGKEY]*ddtarg[sZZ+nSymTen*sYZ] + dg_dtmp[J3KEY]*ddJ3[sZZ+nSymTen*sYZ]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sYZ]*ds[sZZ]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sYZ]*ds[sZZ]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sYZ]*ds[sZZ]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sYZ]*dtarg[sZZ]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sYZ]*dtarg[sZZ]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sYZ]*dtarg[sZZ]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sYZ]*dJ3[sZZ]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sYZ]*dJ3[sZZ]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sYZ]*dJ3[sZZ];

  dr[sZZ+ nSymTen*sZX] = dg_dtmp[TARGKEY]*ddtarg[sZZ+nSymTen*sZX] + dg_dtmp[J3KEY]*ddJ3[sZZ+nSymTen*sZX]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sZX]*ds[sZZ]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sZX]*ds[sZZ]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sZX]*ds[sZZ]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sZX]*dtarg[sZZ]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sZX]*dtarg[sZZ]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sZX]*dtarg[sZZ]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sZX]*dJ3[sZZ]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sZX]*dJ3[sZZ]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sZX]*dJ3[sZZ];

  dr[sZZ+ nSymTen*sXY] = dg_dtmp[TARGKEY]*ddtarg[sZZ+nSymTen*sXY] + dg_dtmp[J3KEY]*ddJ3[sZZ+nSymTen*sXY]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sXY]*ds[sZZ]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sXY]*ds[sZZ]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sXY]*ds[sZZ]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sXY]*dtarg[sZZ]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sXY]*dtarg[sZZ]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sXY]*dtarg[sZZ]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sXY]*dJ3[sZZ]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sXY]*dJ3[sZZ]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sXY]*dJ3[sZZ];

  dr[sYZ+ nSymTen*sXX] = dg_dtmp[TARGKEY]*ddtarg[sYZ+nSymTen*sXX] + dg_dtmp[J3KEY]*ddJ3[sYZ+nSymTen*sXX]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sXX]*ds[sYZ]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sXX]*ds[sYZ]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sXX]*ds[sYZ]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sXX]*dtarg[sYZ]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sXX]*dtarg[sYZ]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sXX]*dtarg[sYZ]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sXX]*dJ3[sYZ]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sXX]*dJ3[sYZ]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sXX]*dJ3[sYZ];

  dr[sYZ+ nSymTen*sYY] = dg_dtmp[TARGKEY]*ddtarg[sYZ+nSymTen*sYY] + dg_dtmp[J3KEY]*ddJ3[sYZ+nSymTen*sYY]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sYY]*ds[sYZ]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sYY]*ds[sYZ]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sYY]*ds[sYZ]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sYY]*dtarg[sYZ]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sYY]*dtarg[sYZ]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sYY]*dtarg[sYZ]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sYY]*dJ3[sYZ]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sYY]*dJ3[sYZ]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sYY]*dJ3[sYZ];

  dr[sYZ+ nSymTen*sZZ] = dg_dtmp[TARGKEY]*ddtarg[sYZ+nSymTen*sZZ] + dg_dtmp[J3KEY]*ddJ3[sYZ+nSymTen*sZZ]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sZZ]*ds[sYZ]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sZZ]*ds[sYZ]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sZZ]*ds[sYZ]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sZZ]*dtarg[sYZ]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sZZ]*dtarg[sYZ]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sZZ]*dtarg[sYZ]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sZZ]*dJ3[sYZ]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sZZ]*dJ3[sYZ]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sZZ]*dJ3[sYZ];

  dr[sYZ+ nSymTen*sYZ] = dg_dtmp[TARGKEY]*ddtarg[sYZ+nSymTen*sYZ] + dg_dtmp[J3KEY]*ddJ3[sYZ+nSymTen*sYZ]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sYZ]*ds[sYZ]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sYZ]*ds[sYZ]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sYZ]*ds[sYZ]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sYZ]*dtarg[sYZ]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sYZ]*dtarg[sYZ]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sYZ]*dtarg[sYZ]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sYZ]*dJ3[sYZ]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sYZ]*dJ3[sYZ]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sYZ]*dJ3[sYZ];

  dr[sYZ+ nSymTen*sZX] = dg_dtmp[TARGKEY]*ddtarg[sYZ+nSymTen*sZX] + dg_dtmp[J3KEY]*ddJ3[sYZ+nSymTen*sZX]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sZX]*ds[sYZ]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sZX]*ds[sYZ]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sZX]*ds[sYZ]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sZX]*dtarg[sYZ]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sZX]*dtarg[sYZ]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sZX]*dtarg[sYZ]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sZX]*dJ3[sYZ]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sZX]*dJ3[sYZ]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sZX]*dJ3[sYZ];

  dr[sYZ+ nSymTen*sXY] = dg_dtmp[TARGKEY]*ddtarg[sYZ+nSymTen*sXY] + dg_dtmp[J3KEY]*ddJ3[sYZ+nSymTen*sXY]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sXY]*ds[sYZ]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sXY]*ds[sYZ]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sXY]*ds[sYZ]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sXY]*dtarg[sYZ]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sXY]*dtarg[sYZ]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sXY]*dtarg[sYZ]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sXY]*dJ3[sYZ]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sXY]*dJ3[sYZ]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sXY]*dJ3[sYZ];

  dr[sZX+ nSymTen*sXX] = dg_dtmp[TARGKEY]*ddtarg[sZX+nSymTen*sXX] + dg_dtmp[J3KEY]*ddJ3[sZX+nSymTen*sXX]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sXX]*ds[sZX]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sXX]*ds[sZX]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sXX]*ds[sZX]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sXX]*dtarg[sZX]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sXX]*dtarg[sZX]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sXX]*dtarg[sZX]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sXX]*dJ3[sZX]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sXX]*dJ3[sZX]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sXX]*dJ3[sZX];

  dr[sZX+ nSymTen*sYY] = dg_dtmp[TARGKEY]*ddtarg[sZX+nSymTen*sYY] + dg_dtmp[J3KEY]*ddJ3[sZX+nSymTen*sYY]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sYY]*ds[sZX]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sYY]*ds[sZX]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sYY]*ds[sZX]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sYY]*dtarg[sZX]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sYY]*dtarg[sZX]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sYY]*dtarg[sZX]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sYY]*dJ3[sZX]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sYY]*dJ3[sZX]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sYY]*dJ3[sZX];

  dr[sZX+ nSymTen*sZZ] = dg_dtmp[TARGKEY]*ddtarg[sZX+nSymTen*sZZ] + dg_dtmp[J3KEY]*ddJ3[sZX+nSymTen*sZZ]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sZZ]*ds[sZX]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sZZ]*ds[sZX]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sZZ]*ds[sZX]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sZZ]*dtarg[sZX]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sZZ]*dtarg[sZX]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sZZ]*dtarg[sZX]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sZZ]*dJ3[sZX]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sZZ]*dJ3[sZX]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sZZ]*dJ3[sZX];

  dr[sZX+ nSymTen*sYZ] = dg_dtmp[TARGKEY]*ddtarg[sZX+nSymTen*sYZ] + dg_dtmp[J3KEY]*ddJ3[sZX+nSymTen*sYZ]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sYZ]*ds[sZX]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sYZ]*ds[sZX]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sYZ]*ds[sZX]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sYZ]*dtarg[sZX]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sYZ]*dtarg[sZX]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sYZ]*dtarg[sZX]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sYZ]*dJ3[sZX]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sYZ]*dJ3[sZX]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sYZ]*dJ3[sZX];

  dr[sZX+ nSymTen*sZX] = dg_dtmp[TARGKEY]*ddtarg[sZX+nSymTen*sZX] + dg_dtmp[J3KEY]*ddJ3[sZX+nSymTen*sZX]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sZX]*ds[sZX]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sZX]*ds[sZX]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sZX]*ds[sZX]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sZX]*dtarg[sZX]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sZX]*dtarg[sZX]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sZX]*dtarg[sZX]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sZX]*dJ3[sZX]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sZX]*dJ3[sZX]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sZX]*dJ3[sZX];

  dr[sZX+ nSymTen*sXY] = dg_dtmp[TARGKEY]*ddtarg[sZX+nSymTen*sXY] + dg_dtmp[J3KEY]*ddJ3[sZX+nSymTen*sXY]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sXY]*ds[sZX]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sXY]*ds[sZX]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sXY]*ds[sZX]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sXY]*dtarg[sZX]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sXY]*dtarg[sZX]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sXY]*dtarg[sZX]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sXY]*dJ3[sZX]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sXY]*dJ3[sZX]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sXY]*dJ3[sZX];

  dr[sXY+ nSymTen*sXX] = dg_dtmp[TARGKEY]*ddtarg[sXY+nSymTen*sXX] + dg_dtmp[J3KEY]*ddJ3[sXY+nSymTen*sXX]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sXX]*ds[sXY]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sXX]*ds[sXY]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sXX]*ds[sXY]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sXX]*dtarg[sXY]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sXX]*dtarg[sXY]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sXX]*dtarg[sXY]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sXX]*dJ3[sXY]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sXX]*dJ3[sXY]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sXX]*dJ3[sXY];

  dr[sXY+ nSymTen*sYY] = dg_dtmp[TARGKEY]*ddtarg[sXY+nSymTen*sYY] + dg_dtmp[J3KEY]*ddJ3[sXY+nSymTen*sYY]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sYY]*ds[sXY]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sYY]*ds[sXY]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sYY]*ds[sXY]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sYY]*dtarg[sXY]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sYY]*dtarg[sXY]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sYY]*dtarg[sXY]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sYY]*dJ3[sXY]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sYY]*dJ3[sXY]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sYY]*dJ3[sXY];

  dr[sXY+ nSymTen*sZZ] = dg_dtmp[TARGKEY]*ddtarg[sXY+nSymTen*sZZ] + dg_dtmp[J3KEY]*ddJ3[sXY+nSymTen*sZZ]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sZZ]*ds[sXY]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sZZ]*ds[sXY]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sZZ]*ds[sXY]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sZZ]*dtarg[sXY]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sZZ]*dtarg[sXY]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sZZ]*dtarg[sXY]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sZZ]*dJ3[sXY]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sZZ]*dJ3[sXY]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sZZ]*dJ3[sXY];

  dr[sXY+ nSymTen*sYZ] = dg_dtmp[TARGKEY]*ddtarg[sXY+nSymTen*sYZ] + dg_dtmp[J3KEY]*ddJ3[sXY+nSymTen*sYZ]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sYZ]*ds[sXY]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sYZ]*ds[sXY]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sYZ]*ds[sXY]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sYZ]*dtarg[sXY]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sYZ]*dtarg[sXY]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sYZ]*dtarg[sXY]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sYZ]*dJ3[sXY]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sYZ]*dJ3[sXY]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sYZ]*dJ3[sXY];

  dr[sXY+ nSymTen*sZX] = dg_dtmp[TARGKEY]*ddtarg[sXY+nSymTen*sZX] + dg_dtmp[J3KEY]*ddJ3[sXY+nSymTen*sZX]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sZX]*ds[sXY]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sZX]*ds[sXY]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sZX]*ds[sXY]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sZX]*dtarg[sXY]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sZX]*dtarg[sXY]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sZX]*dtarg[sXY]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sZX]*dJ3[sXY]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sZX]*dJ3[sXY]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sZX]*dJ3[sXY];

  dr[sXY+ nSymTen*sXY] = dg_dtmp[TARGKEY]*ddtarg[sXY+nSymTen*sXY] + dg_dtmp[J3KEY]*ddJ3[sXY+nSymTen*sXY]\
  + ddg_dtmp[SKEY + 3*SKEY]*ds[sXY]*ds[sXY]  + ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sXY]*ds[sXY]  + ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sXY]*ds[sXY]\
  + ddg_dtmp[TARGKEY + 3*SKEY]*ds[sXY]*dtarg[sXY]  + ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sXY]*dtarg[sXY]  + ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sXY]*dtarg[sXY]\
  + ddg_dtmp[J3KEY + 3*SKEY]*ds[sXY]*dJ3[sXY]  + ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sXY]*dJ3[sXY]  + ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sXY]*dJ3[sXY];

  if (fabs(sin(theta)) >= 0.5-1.0e-8){
    const double sin_theta = copysign(0.5-1.0e-8,sin(theta));
    const double cos_theta = copysign(cos(asin(0.5-1.0e-8)),cos(theta));
  df_dtmp[SKEY] = sqrt(3)*sin(phi)/3;

  df_dtmp[TARGKEY] = sqrt(2)*(-sin_theta*sin(phi)/3 + cos_theta*sqrt(3)/3)/(4*sqrt(targ));

  df_dtmp[J3KEY] = 0;

  df[sXX] = df_dtmp[SKEY]*ds[sXX] + df_dtmp[TARGKEY]*dtarg[sXX] + df_dtmp[J3KEY]*dJ3[sXX];

  df[sYY] = df_dtmp[SKEY]*ds[sYY] + df_dtmp[TARGKEY]*dtarg[sYY] + df_dtmp[J3KEY]*dJ3[sYY];

  df[sZZ] = df_dtmp[SKEY]*ds[sZZ] + df_dtmp[TARGKEY]*dtarg[sZZ] + df_dtmp[J3KEY]*dJ3[sZZ];

  df[sYZ] = df_dtmp[SKEY]*ds[sYZ] + df_dtmp[TARGKEY]*dtarg[sYZ] + df_dtmp[J3KEY]*dJ3[sYZ];

  df[sZX] = df_dtmp[SKEY]*ds[sZX] + df_dtmp[TARGKEY]*dtarg[sZX] + df_dtmp[J3KEY]*dJ3[sZX];

  df[sXY] = df_dtmp[SKEY]*ds[sXY] + df_dtmp[TARGKEY]*dtarg[sXY] + df_dtmp[J3KEY]*dJ3[sXY];

  dg_dtmp[SKEY] = sqrt(3)*sin(psi)/3;

  dg_dtmp[TARGKEY] = sqrt(2)*(-sin_theta*sin(psi)/3 + cos_theta*sqrt(3)/3)/(4*sqrt(targ));

  dg_dtmp[J3KEY] = 0;

  r[sXX] = dg_dtmp[SKEY]*ds[sXX] + dg_dtmp[TARGKEY]*dtarg[sXX] + dg_dtmp[J3KEY]*dJ3[sXX];

  r[sYY] = dg_dtmp[SKEY]*ds[sYY] + dg_dtmp[TARGKEY]*dtarg[sYY] + dg_dtmp[J3KEY]*dJ3[sYY];

  r[sZZ] = dg_dtmp[SKEY]*ds[sZZ] + dg_dtmp[TARGKEY]*dtarg[sZZ] + dg_dtmp[J3KEY]*dJ3[sZZ];

  r[sYZ] = dg_dtmp[SKEY]*ds[sYZ] + dg_dtmp[TARGKEY]*dtarg[sYZ] + dg_dtmp[J3KEY]*dJ3[sYZ];

  r[sZX] = dg_dtmp[SKEY]*ds[sZX] + dg_dtmp[TARGKEY]*dtarg[sZX] + dg_dtmp[J3KEY]*dJ3[sZX];

  r[sXY] = dg_dtmp[SKEY]*ds[sXY] + dg_dtmp[TARGKEY]*dtarg[sXY] + dg_dtmp[J3KEY]*dJ3[sXY];

  ddg_dtmp[SKEY+3*SKEY] = 0;

  ddg_dtmp[SKEY+3*TARGKEY] = 0;

  ddg_dtmp[SKEY+3*J3KEY] = 0;

  ddg_dtmp[TARGKEY+3*SKEY] = 0;

  ddg_dtmp[TARGKEY+3*TARGKEY] = -sqrt(2)*(-sin_theta*sin(psi)/3 + cos_theta*sqrt(3)/3)/(8*pow(targ, 3.0/2.0));

  ddg_dtmp[TARGKEY+3*J3KEY] = 0;

  ddg_dtmp[J3KEY+3*SKEY] = 0;

  ddg_dtmp[J3KEY+3*TARGKEY] = 0;

  ddg_dtmp[J3KEY+3*J3KEY] = 0;

  dr[sXX+ nSymTen*sXX] = dg_dtmp[TARGKEY]*ddtarg[sXX+nSymTen*sXX] + dg_dtmp[J3KEY]*ddJ3[sXX+nSymTen*sXX]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sXX]*ds[sXX]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sXX]*ds[sXX]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sXX]*ds[sXX]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sXX]*dtarg[sXX]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sXX]*dtarg[sXX]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sXX]*dtarg[sXX]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sXX]*dJ3[sXX]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sXX]*dJ3[sXX]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sXX]*dJ3[sXX];

  dr[sXX+ nSymTen*sYY] = dg_dtmp[TARGKEY]*ddtarg[sXX+nSymTen*sYY] + dg_dtmp[J3KEY]*ddJ3[sXX+nSymTen*sYY]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sYY]*ds[sXX]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sYY]*ds[sXX]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sYY]*ds[sXX]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sYY]*dtarg[sXX]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sYY]*dtarg[sXX]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sYY]*dtarg[sXX]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sYY]*dJ3[sXX]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sYY]*dJ3[sXX]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sYY]*dJ3[sXX];

  dr[sXX+ nSymTen*sZZ] = dg_dtmp[TARGKEY]*ddtarg[sXX+nSymTen*sZZ] + dg_dtmp[J3KEY]*ddJ3[sXX+nSymTen*sZZ]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sZZ]*ds[sXX]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sZZ]*ds[sXX]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sZZ]*ds[sXX]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sZZ]*dtarg[sXX]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sZZ]*dtarg[sXX]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sZZ]*dtarg[sXX]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sZZ]*dJ3[sXX]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sZZ]*dJ3[sXX]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sZZ]*dJ3[sXX];

  dr[sXX+ nSymTen*sYZ] = dg_dtmp[TARGKEY]*ddtarg[sXX+nSymTen*sYZ] + dg_dtmp[J3KEY]*ddJ3[sXX+nSymTen*sYZ]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sYZ]*ds[sXX]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sYZ]*ds[sXX]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sYZ]*ds[sXX]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sYZ]*dtarg[sXX]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sYZ]*dtarg[sXX]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sYZ]*dtarg[sXX]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sYZ]*dJ3[sXX]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sYZ]*dJ3[sXX]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sYZ]*dJ3[sXX];

  dr[sXX+ nSymTen*sZX] = dg_dtmp[TARGKEY]*ddtarg[sXX+nSymTen*sZX] + dg_dtmp[J3KEY]*ddJ3[sXX+nSymTen*sZX]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sZX]*ds[sXX]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sZX]*ds[sXX]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sZX]*ds[sXX]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sZX]*dtarg[sXX]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sZX]*dtarg[sXX]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sZX]*dtarg[sXX]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sZX]*dJ3[sXX]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sZX]*dJ3[sXX]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sZX]*dJ3[sXX];

  dr[sXX+ nSymTen*sXY] = dg_dtmp[TARGKEY]*ddtarg[sXX+nSymTen*sXY] + dg_dtmp[J3KEY]*ddJ3[sXX+nSymTen*sXY]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sXY]*ds[sXX]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sXY]*ds[sXX]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sXY]*ds[sXX]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sXY]*dtarg[sXX]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sXY]*dtarg[sXX]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sXY]*dtarg[sXX]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sXY]*dJ3[sXX]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sXY]*dJ3[sXX]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sXY]*dJ3[sXX];

  dr[sYY+ nSymTen*sXX] = dg_dtmp[TARGKEY]*ddtarg[sYY+nSymTen*sXX] + dg_dtmp[J3KEY]*ddJ3[sYY+nSymTen*sXX]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sXX]*ds[sYY]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sXX]*ds[sYY]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sXX]*ds[sYY]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sXX]*dtarg[sYY]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sXX]*dtarg[sYY]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sXX]*dtarg[sYY]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sXX]*dJ3[sYY]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sXX]*dJ3[sYY]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sXX]*dJ3[sYY];

  dr[sYY+ nSymTen*sYY] = dg_dtmp[TARGKEY]*ddtarg[sYY+nSymTen*sYY] + dg_dtmp[J3KEY]*ddJ3[sYY+nSymTen*sYY]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sYY]*ds[sYY]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sYY]*ds[sYY]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sYY]*ds[sYY]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sYY]*dtarg[sYY]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sYY]*dtarg[sYY]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sYY]*dtarg[sYY]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sYY]*dJ3[sYY]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sYY]*dJ3[sYY]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sYY]*dJ3[sYY];

  dr[sYY+ nSymTen*sZZ] = dg_dtmp[TARGKEY]*ddtarg[sYY+nSymTen*sZZ] + dg_dtmp[J3KEY]*ddJ3[sYY+nSymTen*sZZ]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sZZ]*ds[sYY]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sZZ]*ds[sYY]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sZZ]*ds[sYY]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sZZ]*dtarg[sYY]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sZZ]*dtarg[sYY]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sZZ]*dtarg[sYY]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sZZ]*dJ3[sYY]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sZZ]*dJ3[sYY]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sZZ]*dJ3[sYY];

  dr[sYY+ nSymTen*sYZ] = dg_dtmp[TARGKEY]*ddtarg[sYY+nSymTen*sYZ] + dg_dtmp[J3KEY]*ddJ3[sYY+nSymTen*sYZ]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sYZ]*ds[sYY]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sYZ]*ds[sYY]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sYZ]*ds[sYY]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sYZ]*dtarg[sYY]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sYZ]*dtarg[sYY]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sYZ]*dtarg[sYY]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sYZ]*dJ3[sYY]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sYZ]*dJ3[sYY]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sYZ]*dJ3[sYY];

  dr[sYY+ nSymTen*sZX] = dg_dtmp[TARGKEY]*ddtarg[sYY+nSymTen*sZX] + dg_dtmp[J3KEY]*ddJ3[sYY+nSymTen*sZX]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sZX]*ds[sYY]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sZX]*ds[sYY]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sZX]*ds[sYY]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sZX]*dtarg[sYY]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sZX]*dtarg[sYY]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sZX]*dtarg[sYY]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sZX]*dJ3[sYY]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sZX]*dJ3[sYY]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sZX]*dJ3[sYY];

  dr[sYY+ nSymTen*sXY] = dg_dtmp[TARGKEY]*ddtarg[sYY+nSymTen*sXY] + dg_dtmp[J3KEY]*ddJ3[sYY+nSymTen*sXY]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sXY]*ds[sYY]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sXY]*ds[sYY]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sXY]*ds[sYY]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sXY]*dtarg[sYY]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sXY]*dtarg[sYY]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sXY]*dtarg[sYY]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sXY]*dJ3[sYY]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sXY]*dJ3[sYY]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sXY]*dJ3[sYY];

  dr[sZZ+ nSymTen*sXX] = dg_dtmp[TARGKEY]*ddtarg[sZZ+nSymTen*sXX] + dg_dtmp[J3KEY]*ddJ3[sZZ+nSymTen*sXX]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sXX]*ds[sZZ]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sXX]*ds[sZZ]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sXX]*ds[sZZ]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sXX]*dtarg[sZZ]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sXX]*dtarg[sZZ]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sXX]*dtarg[sZZ]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sXX]*dJ3[sZZ]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sXX]*dJ3[sZZ]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sXX]*dJ3[sZZ];

  dr[sZZ+ nSymTen*sYY] = dg_dtmp[TARGKEY]*ddtarg[sZZ+nSymTen*sYY] + dg_dtmp[J3KEY]*ddJ3[sZZ+nSymTen*sYY]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sYY]*ds[sZZ]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sYY]*ds[sZZ]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sYY]*ds[sZZ]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sYY]*dtarg[sZZ]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sYY]*dtarg[sZZ]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sYY]*dtarg[sZZ]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sYY]*dJ3[sZZ]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sYY]*dJ3[sZZ]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sYY]*dJ3[sZZ];

  dr[sZZ+ nSymTen*sZZ] = dg_dtmp[TARGKEY]*ddtarg[sZZ+nSymTen*sZZ] + dg_dtmp[J3KEY]*ddJ3[sZZ+nSymTen*sZZ]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sZZ]*ds[sZZ]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sZZ]*ds[sZZ]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sZZ]*ds[sZZ]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sZZ]*dtarg[sZZ]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sZZ]*dtarg[sZZ]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sZZ]*dtarg[sZZ]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sZZ]*dJ3[sZZ]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sZZ]*dJ3[sZZ]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sZZ]*dJ3[sZZ];

  dr[sZZ+ nSymTen*sYZ] = dg_dtmp[TARGKEY]*ddtarg[sZZ+nSymTen*sYZ] + dg_dtmp[J3KEY]*ddJ3[sZZ+nSymTen*sYZ]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sYZ]*ds[sZZ]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sYZ]*ds[sZZ]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sYZ]*ds[sZZ]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sYZ]*dtarg[sZZ]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sYZ]*dtarg[sZZ]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sYZ]*dtarg[sZZ]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sYZ]*dJ3[sZZ]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sYZ]*dJ3[sZZ]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sYZ]*dJ3[sZZ];

  dr[sZZ+ nSymTen*sZX] = dg_dtmp[TARGKEY]*ddtarg[sZZ+nSymTen*sZX] + dg_dtmp[J3KEY]*ddJ3[sZZ+nSymTen*sZX]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sZX]*ds[sZZ]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sZX]*ds[sZZ]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sZX]*ds[sZZ]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sZX]*dtarg[sZZ]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sZX]*dtarg[sZZ]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sZX]*dtarg[sZZ]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sZX]*dJ3[sZZ]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sZX]*dJ3[sZZ]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sZX]*dJ3[sZZ];

  dr[sZZ+ nSymTen*sXY] = dg_dtmp[TARGKEY]*ddtarg[sZZ+nSymTen*sXY] + dg_dtmp[J3KEY]*ddJ3[sZZ+nSymTen*sXY]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sXY]*ds[sZZ]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sXY]*ds[sZZ]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sXY]*ds[sZZ]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sXY]*dtarg[sZZ]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sXY]*dtarg[sZZ]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sXY]*dtarg[sZZ]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sXY]*dJ3[sZZ]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sXY]*dJ3[sZZ]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sXY]*dJ3[sZZ];

  dr[sYZ+ nSymTen*sXX] = dg_dtmp[TARGKEY]*ddtarg[sYZ+nSymTen*sXX] + dg_dtmp[J3KEY]*ddJ3[sYZ+nSymTen*sXX]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sXX]*ds[sYZ]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sXX]*ds[sYZ]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sXX]*ds[sYZ]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sXX]*dtarg[sYZ]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sXX]*dtarg[sYZ]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sXX]*dtarg[sYZ]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sXX]*dJ3[sYZ]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sXX]*dJ3[sYZ]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sXX]*dJ3[sYZ];

  dr[sYZ+ nSymTen*sYY] = dg_dtmp[TARGKEY]*ddtarg[sYZ+nSymTen*sYY] + dg_dtmp[J3KEY]*ddJ3[sYZ+nSymTen*sYY]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sYY]*ds[sYZ]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sYY]*ds[sYZ]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sYY]*ds[sYZ]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sYY]*dtarg[sYZ]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sYY]*dtarg[sYZ]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sYY]*dtarg[sYZ]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sYY]*dJ3[sYZ]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sYY]*dJ3[sYZ]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sYY]*dJ3[sYZ];

  dr[sYZ+ nSymTen*sZZ] = dg_dtmp[TARGKEY]*ddtarg[sYZ+nSymTen*sZZ] + dg_dtmp[J3KEY]*ddJ3[sYZ+nSymTen*sZZ]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sZZ]*ds[sYZ]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sZZ]*ds[sYZ]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sZZ]*ds[sYZ]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sZZ]*dtarg[sYZ]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sZZ]*dtarg[sYZ]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sZZ]*dtarg[sYZ]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sZZ]*dJ3[sYZ]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sZZ]*dJ3[sYZ]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sZZ]*dJ3[sYZ];

  dr[sYZ+ nSymTen*sYZ] = dg_dtmp[TARGKEY]*ddtarg[sYZ+nSymTen*sYZ] + dg_dtmp[J3KEY]*ddJ3[sYZ+nSymTen*sYZ]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sYZ]*ds[sYZ]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sYZ]*ds[sYZ]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sYZ]*ds[sYZ]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sYZ]*dtarg[sYZ]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sYZ]*dtarg[sYZ]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sYZ]*dtarg[sYZ]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sYZ]*dJ3[sYZ]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sYZ]*dJ3[sYZ]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sYZ]*dJ3[sYZ];

  dr[sYZ+ nSymTen*sZX] = dg_dtmp[TARGKEY]*ddtarg[sYZ+nSymTen*sZX] + dg_dtmp[J3KEY]*ddJ3[sYZ+nSymTen*sZX]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sZX]*ds[sYZ]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sZX]*ds[sYZ]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sZX]*ds[sYZ]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sZX]*dtarg[sYZ]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sZX]*dtarg[sYZ]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sZX]*dtarg[sYZ]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sZX]*dJ3[sYZ]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sZX]*dJ3[sYZ]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sZX]*dJ3[sYZ];

  dr[sYZ+ nSymTen*sXY] = dg_dtmp[TARGKEY]*ddtarg[sYZ+nSymTen*sXY] + dg_dtmp[J3KEY]*ddJ3[sYZ+nSymTen*sXY]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sXY]*ds[sYZ]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sXY]*ds[sYZ]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sXY]*ds[sYZ]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sXY]*dtarg[sYZ]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sXY]*dtarg[sYZ]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sXY]*dtarg[sYZ]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sXY]*dJ3[sYZ]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sXY]*dJ3[sYZ]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sXY]*dJ3[sYZ];

  dr[sZX+ nSymTen*sXX] = dg_dtmp[TARGKEY]*ddtarg[sZX+nSymTen*sXX] + dg_dtmp[J3KEY]*ddJ3[sZX+nSymTen*sXX]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sXX]*ds[sZX]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sXX]*ds[sZX]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sXX]*ds[sZX]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sXX]*dtarg[sZX]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sXX]*dtarg[sZX]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sXX]*dtarg[sZX]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sXX]*dJ3[sZX]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sXX]*dJ3[sZX]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sXX]*dJ3[sZX];

  dr[sZX+ nSymTen*sYY] = dg_dtmp[TARGKEY]*ddtarg[sZX+nSymTen*sYY] + dg_dtmp[J3KEY]*ddJ3[sZX+nSymTen*sYY]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sYY]*ds[sZX]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sYY]*ds[sZX]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sYY]*ds[sZX]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sYY]*dtarg[sZX]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sYY]*dtarg[sZX]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sYY]*dtarg[sZX]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sYY]*dJ3[sZX]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sYY]*dJ3[sZX]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sYY]*dJ3[sZX];

  dr[sZX+ nSymTen*sZZ] = dg_dtmp[TARGKEY]*ddtarg[sZX+nSymTen*sZZ] + dg_dtmp[J3KEY]*ddJ3[sZX+nSymTen*sZZ]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sZZ]*ds[sZX]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sZZ]*ds[sZX]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sZZ]*ds[sZX]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sZZ]*dtarg[sZX]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sZZ]*dtarg[sZX]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sZZ]*dtarg[sZX]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sZZ]*dJ3[sZX]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sZZ]*dJ3[sZX]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sZZ]*dJ3[sZX];

  dr[sZX+ nSymTen*sYZ] = dg_dtmp[TARGKEY]*ddtarg[sZX+nSymTen*sYZ] + dg_dtmp[J3KEY]*ddJ3[sZX+nSymTen*sYZ]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sYZ]*ds[sZX]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sYZ]*ds[sZX]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sYZ]*ds[sZX]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sYZ]*dtarg[sZX]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sYZ]*dtarg[sZX]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sYZ]*dtarg[sZX]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sYZ]*dJ3[sZX]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sYZ]*dJ3[sZX]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sYZ]*dJ3[sZX];

  dr[sZX+ nSymTen*sZX] = dg_dtmp[TARGKEY]*ddtarg[sZX+nSymTen*sZX] + dg_dtmp[J3KEY]*ddJ3[sZX+nSymTen*sZX]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sZX]*ds[sZX]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sZX]*ds[sZX]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sZX]*ds[sZX]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sZX]*dtarg[sZX]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sZX]*dtarg[sZX]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sZX]*dtarg[sZX]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sZX]*dJ3[sZX]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sZX]*dJ3[sZX]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sZX]*dJ3[sZX];

  dr[sZX+ nSymTen*sXY] = dg_dtmp[TARGKEY]*ddtarg[sZX+nSymTen*sXY] + dg_dtmp[J3KEY]*ddJ3[sZX+nSymTen*sXY]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sXY]*ds[sZX]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sXY]*ds[sZX]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sXY]*ds[sZX]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sXY]*dtarg[sZX]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sXY]*dtarg[sZX]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sXY]*dtarg[sZX]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sXY]*dJ3[sZX]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sXY]*dJ3[sZX]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sXY]*dJ3[sZX];

  dr[sXY+ nSymTen*sXX] = dg_dtmp[TARGKEY]*ddtarg[sXY+nSymTen*sXX] + dg_dtmp[J3KEY]*ddJ3[sXY+nSymTen*sXX]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sXX]*ds[sXY]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sXX]*ds[sXY]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sXX]*ds[sXY]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sXX]*dtarg[sXY]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sXX]*dtarg[sXY]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sXX]*dtarg[sXY]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sXX]*dJ3[sXY]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sXX]*dJ3[sXY]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sXX]*dJ3[sXY];

  dr[sXY+ nSymTen*sYY] = dg_dtmp[TARGKEY]*ddtarg[sXY+nSymTen*sYY] + dg_dtmp[J3KEY]*ddJ3[sXY+nSymTen*sYY]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sYY]*ds[sXY]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sYY]*ds[sXY]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sYY]*ds[sXY]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sYY]*dtarg[sXY]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sYY]*dtarg[sXY]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sYY]*dtarg[sXY]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sYY]*dJ3[sXY]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sYY]*dJ3[sXY]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sYY]*dJ3[sXY];

  dr[sXY+ nSymTen*sZZ] = dg_dtmp[TARGKEY]*ddtarg[sXY+nSymTen*sZZ] + dg_dtmp[J3KEY]*ddJ3[sXY+nSymTen*sZZ]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sZZ]*ds[sXY]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sZZ]*ds[sXY]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sZZ]*ds[sXY]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sZZ]*dtarg[sXY]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sZZ]*dtarg[sXY]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sZZ]*dtarg[sXY]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sZZ]*dJ3[sXY]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sZZ]*dJ3[sXY]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sZZ]*dJ3[sXY];

  dr[sXY+ nSymTen*sYZ] = dg_dtmp[TARGKEY]*ddtarg[sXY+nSymTen*sYZ] + dg_dtmp[J3KEY]*ddJ3[sXY+nSymTen*sYZ]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sYZ]*ds[sXY]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sYZ]*ds[sXY]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sYZ]*ds[sXY]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sYZ]*dtarg[sXY]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sYZ]*dtarg[sXY]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sYZ]*dtarg[sXY]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sYZ]*dJ3[sXY]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sYZ]*dJ3[sXY]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sYZ]*dJ3[sXY];

  dr[sXY+ nSymTen*sZX] = dg_dtmp[TARGKEY]*ddtarg[sXY+nSymTen*sZX] + dg_dtmp[J3KEY]*ddJ3[sXY+nSymTen*sZX]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sZX]*ds[sXY]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sZX]*ds[sXY]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sZX]*ds[sXY]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sZX]*dtarg[sXY]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sZX]*dtarg[sXY]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sZX]*dtarg[sXY]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sZX]*dJ3[sXY]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sZX]*dJ3[sXY]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sZX]*dJ3[sXY];

  dr[sXY+ nSymTen*sXY] = dg_dtmp[TARGKEY]*ddtarg[sXY+nSymTen*sXY] + dg_dtmp[J3KEY]*ddJ3[sXY+nSymTen*sXY]\
+ ddg_dtmp[SKEY + 3*SKEY]*ds[sXY]*ds[sXY]+ ddg_dtmp[SKEY + 3*TARGKEY]*dtarg[sXY]*ds[sXY]+ ddg_dtmp[SKEY + 3*J3KEY]*dJ3[sXY]*ds[sXY]\
+ ddg_dtmp[TARGKEY + 3*SKEY]*ds[sXY]*dtarg[sXY]+ ddg_dtmp[TARGKEY + 3*TARGKEY]*dtarg[sXY]*dtarg[sXY]+ ddg_dtmp[TARGKEY + 3*J3KEY]*dJ3[sXY]*dtarg[sXY]\
+ ddg_dtmp[J3KEY + 3*SKEY]*ds[sXY]*dJ3[sXY]+ ddg_dtmp[J3KEY + 3*TARGKEY]*dtarg[sXY]*dJ3[sXY]+ ddg_dtmp[J3KEY + 3*J3KEY]*dJ3[sXY]*dJ3[sXY];

}
}
