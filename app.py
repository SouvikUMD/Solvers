# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 16:33:58 2024

@author: prama
"""

import numpy as np
import math
import io
from flask import Flask, render_template, request, Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


app = Flask(__name__)


# Helper Functions.
"""Eigenvector Eignevalue Solver: Solves for Eignenvectors and Eigenvalues. 
The NumPy function np.linalg.elg() was used."""
def create_figure(D, Dn, WB, NA, VT, VA, ni):
    q = 1.6e-19  # Charge of electron
    A = (D**2)*1e-12 # BJT Cross-Sectional Area converted to m^2
    # Dn = 20 cm^2/s (m^2)
    # 0.5 um
    npo = (ni**2/NA)*1e6   # npo = ni^2/NA (/m^3)

    # VCE represents the collector-emitter voltage
    VCE = np.arange(0, 5, 0.001)
    # VBE = 0.4 V
    IC1 = ((q*A*Dn)/WB)*npo*math.exp(0.4/VT)*(1 + VCE/VA);
    fig = plt.figure(figure=(16,6))
    plt.plot(VCE, IC1)
    #VBE = 0.5 V
    IC2 = ((q*A*Dn)/WB)*npo*math.exp(0.5/VT)*(1 + VCE/VA);
    plt.plot(VCE, IC2)
    # VBE = 0.6 V
    IC3 = ((q*A*Dn)/WB)*npo*math.exp(0.6/VT)*(1 + VCE/VA);
    plt.plot(VCE, IC3)
    plt.xlabel('VCE(V)')
    plt.ylabel('IC(V)')
    plt.title('Collector Current vs. Collector-Emitter Voltage')
    plt.legend(['Vbe = 0.4 V','Vbe = 0.5 V','Vbe = 0.6 V'])
    return q, A, npo, fig

def display_info(q, A, npo, fig):
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

"""Eigenvector Eignevalue Solver: Solves for Eignenvectors and Eigenvalues. 
The NumPy function np.linalg.elg() was used."""

def rotate_arbitrary(x, y, x0, y0, theta, ax):
  ''' Computes coordinates from source (x,y) to destination (x', y') by rotating
  about an axis (x/y/z) a point about an arbitrary reference location (x0, y0).
  Input:
    (x, y): source coordinate.
    (x0, y0): reference location.
    theta: Angle of rotation.
    ax: axis(x, y, or z)
  Output:
    xtilda_vec: destination points (Will be in homogeneous form. Return x and
    y coordinates only.)
  '''
  # Express (x,y) in homogeneous form xvector.
  xvector = np.mat([x, y, 1.0])
  xvector = np.transpose(xvector)
  # Convert theta to radians (theta_rad).
  theta_rad = float(theta) * np.pi/180
  # Rotation Matrix
  # Rz: Rotation Matrix about the z-axis
  if (ax.lower() == 'z') or ('z-axis' in ax):
    R = np.mat([[np.cos(theta_rad), -np.sin(theta_rad), 0.0],
              [np.sin(theta_rad), np.cos(theta_rad), 0.0],
              [0.0, 0.0, 1.0]])
  elif (ax.lower() == 'x') or ('x-axis' in ax):
    R = np.mat([[1.0, 0.0, 0.0],
            [0.0, np.cos(theta_rad), -np.sin(theta_rad)],
            [0.0, np.sin(theta_rad), np.cos(theta_rad)]])
  elif (ax.lower() == 'y') or ('y-axis' in ax):
    R = np.mat([[np.cos(theta_rad), 0.0, np.sin(theta_rad)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta_rad), 0.0, np.cos(theta_rad)]])
  else:
    R = np.mat([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

  #T_pos_theta: translation matrix by (x0, y0).
  T_pos_theta = np.mat([[1.0, 0.0, x0], [0.0, 1.0, y0], [0.0, 0.0, 1.0]])
  #T_neg_theta: Translation Matrix by (-x0, -y0).
  T_neg_theta = np.mat([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]])
  #print("T_neg_theta: ", T_neg_theta)
  # Perform matrix multiplication between T+, R_x/R_y/R_z, T-, and xvector.
  xtilda_vec = T_pos_theta @ R @ T_neg_theta @ xvector
  #xtilda_vec = T_pos_theta @ R_z @ T_neg_theta @ xvector
  #xtilda_vec = T_pos_theta @ R_x @ T_neg_theta @ xvector
  #xtilda_vec = T_pos_theta @ R_y @ T_neg_theta @ xvector
  # Set xtilda_vec to 1D vector.
  xtilda_vec = np.ravel(xtilda_vec)
  # Return (x', y'). x' = xtilda_vec[0], y' = xtilda_vec[1].
  return xtilda_vec[0], xtilda_vec[1]


"""Compute and Apply Homography."""

def compute_homography(src, dst):
  '''Computes the homography from src to dst.
   Input:
    src: source points, shape (N, 2), where N >= 4
    dst: destination points, shape (N, 2)
   Output:
    H: homography from source points to destination points, shape (3, 3)
  '''

  H = np.identity(3) # Create identity matrix of H.
  if (src.shape[1] != 2) or (dst.shape[1] != 2) or (len(src) != len(dst)):
    print("Will not be calculated due to mismatched shapes. \
          Please click the back button and try again.")
    return H

  # Need to derive Q. From Q*a = 0.
  Q = []
  for i in range(len(src)):
      # Creating 2Nx9 matrix where N is the number of two-dimensional points.
      # Derived from Problem 1a which is based on Packet 9 Slides 9-15.
      xi = src[i, 0]
      yi = src[i, 1]
      xip = dst[i, 0]
      yip = dst[i, 1]
      row1 = [xi, yi, 1, 0, 0, 0, -xi*xip, -yi*xip, -xip]
      Q.append(row1)
      row2 = [0, 0, 0, xi, yi, 1, -xi*yip, -yi*yip, -yip]
      Q.append(row2)

  Q = np.array(Q)
  Qtranspose = Q.transpose()  # Need to transpose Q
  M = np.matmul(Qtranspose, Q)  # Matrix multiplication of transpose Q and Q.
  eigvalQ, eigvecQ = np.linalg.eigh(M)
  mymin = np.min(eigvalQ) # Smallest eigvalQ.
  min_position = np.argmin(eigvalQ)
  H = eigvecQ[:, min_position]
  H = H.reshape(3, 3) # Reshape H to (3x3) array.

  return H


def apply_homography(src, H):
  '''Applies a homography H to the source points.
   Input:
      src: source points, shape (N, 2)
      H: homography from source points to destination points, shape (3, 3)
   Output:
     dst: destination points, shape (N, 2)
  '''
  # The following line is temporary. Replace it with your code.
  dst = np.zeros([4, 2])
  src = np.array(src)

  if (len(src[0]) != 2) or (len(H) != 3) or (len(H[0]) != 3):
    return dst

  dst = []
  for i in range(len(src)):
    point = [src[i, 0], src[i, 1], 1] # Need a scalar value of 1.
    point = np.array(point)
    # Multiply H*(source points)
    newloc = np.matmul(H, np.transpose(point))
    # normalize newloc
    norm_elt = newloc[len(newloc) - 1]
    for i in range(len(newloc)):
      newloc[i] = newloc[i]/norm_elt

    # Take first two elements in newloc and append to dst.
    newloc_pts = [newloc[0], newloc[1]]

    dst.append(newloc_pts)

  dst = np.array(dst)

  return dst

# Implement Flask on app.

# In case input is provided incorrectly.
@app.route('/error_msg')
def error_msg_text():
    return render_template("try_again.html")

# Eigenvalue/Eigenvector Solver.
@app.route('/eigvalue_eigvector_solver')
def upload_form_eig_solver():
    return render_template("Eigenvalue_Eignvector.html")


@app.route('/eigvalue_eigvector_solver', methods=['GET', 'POST'])
def eigvalue_eigvector_solver():
  Q = request.form.get("qmatrix")
  try:
      M_var = np.matrix(Q)
      if M_var.shape[1] == 0:
          return "Blank Matrix provided.Please click the back\
              button and try again."
  except ValueError:
      return "Cannot convert provided input into matrix. Please click the back\
          button and try again."
  if M_var.shape[0] != M_var.shape[1]:
    Mtranspose = M_var.transpose()  # Q^T
    M_var = np.matmul(Mtranspose, M_var)  # M = Q^T*Q.


  # Used to find eigenvalues (eigvalQ) and eignevectors (eigvecQ).
  eigvalQ, eigvecQ = np.linalg.eigh(M_var)

  # mymin: Smallest eigvalQ.
  mymin = np.min(eigvalQ)
  min_position = np.argmin(eigvalQ) # Obtain position of mymin.
  H = eigvecQ[:, min_position] # Print eignevectors of mymin.
  display_eigvalQ = f'eigvalQ:  {eigvalQ}' + '<br/>'
  display_eigvecQ = f'eigvecQ:  {eigvecQ}' + '<br/>'
  disp_small_eigvecQ = f"Smallest eigvalQ: {mymin}" + '<br/>'
  disp_H = f"H: {np.transpose(H)}"
  eigval_summary = display_eigvalQ +  display_eigvecQ + disp_small_eigvecQ + disp_H
  return eigval_summary


@app.route('/arbitary_pt_solver')
def upload_form_arbitrary_pt():
    return render_template("Arbitrary_Point_Solver.html")

@app.route('/arbitary_pt_solver', methods=['GET', 'POST'])
def select_pts_for_arbitrary_pt_solver():
  x= request.form.get("xcoord")
  try: 
      x = float(x)
  except: 
      return error_msg_text()
  y= request.form.get("ycoord")
  try:
      y = float(y)
  except:
      return error_msg_text()
  x0 = request.form.get("xarb")
  try:
      x0 = float(x0)
  except:
      return error_msg_text()
  y0 = request.form.get("yarb")
  try:
      y0 = float(y0)
  except:
      return error_msg_text()
  deg = request.form.get("angle")

  try:
      deg = float(deg)
  except:
      return error_msg_text()
  axis_sel = request.form.get("axis")
  if (not axis_sel.lower() == 'x') and (not axis_sel.lower() == 'x-axis') \
      and (not axis_sel.lower() == 'y') and (not axis_sel.lower() == 'y-axis') \
      and (not axis_sel.lower() == 'z') and (not axis_sel.lower() == 'z-axis'):
          return error_msg_text()
          
  x_dst, y_dst = rotate_arbitrary(x, y, x0, y0, deg, axis_sel)
  print_src = f'src_coordinates (x, y): ({str(x)}, {str(y)})'
  print_arb = f'arbitrary points (x0, y0): ({str(x0)}, {str(y0)})'
  print_deg = f'{str(deg)} degrees'
  print_dst = f'''dst_coordinates (x', y'): ({str(x_dst)}, {str(y_dst)})'''
  summary= print_src + '<br/>' +  print_arb + '<br/>' + print_deg + '<br/>' + \
      print_dst
  return summary

@app.route('/homography')
def upload_form_homography():
    return render_template("Homography.html")

@app.route('/homography', methods=['GET', 'POST'])
def select_pts_homography():
  #print("Enter Source Points: ")
  #src_pts = input()
  src_pts = request.form.get("src_coord")
  try:
      src_pts = np.matrix(src_pts)
      if src_pts.shape[1] == 0:
          return "Blank Matrix provided.Please click the back\
              button and try again."
  except ValueError:
      return error_msg_text()
      
  disp_src_pts = f'src_pts: {src_pts}' + '<br/>'
  #print("Enter Destination Points: ")
  #dst_pts = input()
  dst_pts = request.form.get("dst_coord")
  try:
      dst_pts = np.matrix(dst_pts)
      if dst_pts.shape[1] == 0:
          return "Blank Matrix provided.Please click the back\
              button and try again."
  except ValueError:
      return error_msg_text()

  #print(type(dst_pts))
  disp_dst_pts = f'dst_pts: {dst_pts}' + '<br/>'
  H = compute_homography(src_pts, dst_pts)
  disp_H = f'H(Homography): {H}' + '<br/>'
  match_pts = apply_homography(src_pts, H)
  disp_match_pts = f'match_pts: {match_pts}' + '<br/>'
  diff = np.square(match_pts - dst_pts).sum()
  disp_diff = f'Your 3rd solution differs from our solution by: {diff}'

  homography_summary = disp_src_pts +  disp_dst_pts + disp_H + disp_diff
  return homography_summary

@app.route('/VCE_BJT_Form')
def upload_form_VCE_BJT():
    return render_template("VCE_BJT_Form.html")

@app.route('/VCE_BJT_Form', methods=['GET', 'POST'])
def enter_VCE_BJT_info():
  D = request.form.get("dimension")
  try:
      D = float(D)
  except:
      return error_msg_text()

  Dn = request.form.get("Dn")
  try:
      Dn = float(Dn)
  except:
      return error_msg_text()
  
  WB = request.form.get("WB")
  try:
      WB = float(WB)
  except:
      return error_msg_text()
  
  NA = request.form.get("NA")

  try:
      NA = float(NA)
  except:
      return error_msg_text()
  
  VT = request.form.get("Voltage Transformer")
  try:
      VT = float(VT)
  except:
      return error_msg_text()
  
  VA = request.form.get("Voltage A")
  try:
      VA = float(VA)
  except:
      return error_msg_text()
  
  ni = request.form.get("ni")
  try:
      ni = float(ni)
  except:
      return error_msg_text()
  
  q, A, npo, fig = create_figure(D, Dn, WB, NA, VT, VA, ni)
  return display_info(q, A, npo, fig)

@app.route('/')
def home():
    return render_template("home.html")

if __name__ == '__main__':
    app.run()
