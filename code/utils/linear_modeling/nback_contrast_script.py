from scipy import stats

img1 = nib.load("../../../data/sub001/BOLD/task003_run001/filtered_func_data_mni.nii.gz")
img = nib.load('../../../data/' + f1 + '.nii.gz')
data1 = img1.get_data()

######### Get n_trs and voxel shape
n_trs1 = data1.shape[-1]
vol_shape1 = data1.shape[:-1]

######### Construct design matrix:
# Load the convolution files.
design_mat = np.ones((n_trs, 9))
files = ('1', '2', '3', '4', '5', '6')
for file in files:
    design_mat[:, (int(file) - 1)] = np.loadtxt('../../../data/convo_prep/task001_run001_cond00'
                                          + file + '_conv.txt')
# adding the linear drifter terms to the design matrix
linear_drift = np.linspace(-1, 1, n_trs)
design_mat[:, 6] = linear_drift
quadratic_drift = linear_drift ** 2
quadratic_drift -= np.mean(quadratic_drift)
design_mat[:, 7] = quadratic_drift
# show the design matrix graphically:
plt.imshow(design_mat, aspect=0.1)
#plt.show()
#plt.savefig('../../../data/design_matrix/design_mat.png')


######### we take the mean volume (over time), and do a histogram of the values
mean_vol1 = np.mean(data1, axis=-1)
plt.hist(np.ravel(mean_vol), bins=100)
plt.xlabel('Voxels')
plt.ylabel('Frequency')
plt.title('Mean Volume Over Time')
plt.show()
#plt.savefig("../../../data/design_matrix/mean_vol.png")
# mask out the outer-brain noise using mean volumes over time.
in_brain_mask1 = mean_vol1 > 5000
# We can use this 3D mask to index into our 4D dataset.
# This selects all the voxel time-courses for voxels within the brain
# (as defined by the mask)
in_brain_tcs1 = data1[in_brain_mask1, :]



######### Lastly, do t test on betas:
y1 = in_brain_tcs1.T
X = design_mat

beta1, MRSS1, df1 = linear_modeling.beta_est(y1,X)

# Visualizing betas for the middle slice
# First reshape
b_vols1 = linear_modeling.reshape(in_brain_mask1, vol_shape, beta1)
# Then plot them on the same plot with uniform scale
fig, axes = plt.subplots(nrows=2, ncols=4)
for i, ax in zip(range(0,8,1), axes.flat):
    im = ax.imshow(b_vols[:, :, 45, i], cmap = 'jet')
fig.subplots_adjust(right=0.85)
cax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
fig.colorbar(im, cax=cax)
#plt.show()

# To test significance of betas:
# Create contrast matrix for each beta:
c_mat = np.diag(np.array(np.ones((9,))))
# t statistics and p values
# Length is the number of voxels after masking
t_mat = np.ones((9, y.shape[1]))
p_mat = np.ones((9, y.shape[1],))
for i in range(0,9,1):
    t, p = linear_modeling.t_stat(X, c_mat[:,i], beta, MRSS, df)
    t_mat[i,:] = t
    p_mat[i,:] = p
# save the t values and p values in txt files.
#np.savetxt('../../../data/maps/t_stat.txt', t_mat)
#np.savetxt('../../../data/maps/p_val.txt', p_mat)
t_val = np.zeros(vol_shape + (t_mat.shape[0],))
t_val[in_brain_mask, :] = t_mat.T
# Reshape p values
p_val = np.ones(vol_shape + (p_mat.shape[0],))
p_val[in_brain_mask, :] = p_mat.T


# n_back contrast
t_values, p_values = stats.ttest_ind(beta1, beta, axis = 1, equal_var = False)


# smoothing
fmri_img = image.smooth_img('../../../data/sub001/BOLD/task001_run001/filtered_func_data_mni.nii.gz', fwhm=6)
mean_img = image.mean_img(fmri_img)
# Thresholding
p_val = np.ones(vol_shape + (p_mat.shape[0],))
p_val[in_brain_mask, :] = p_mat.T

log_p_values = -np.log10(p_val[..., 5])
log_p_values[np.isnan(log_p_values)] = 0.
log_p_values[log_p_values > 10.] = 10.
log_p_values[log_p_values < -np.log10(0.05/137)] = 0
plot_stat_map(nibabel.Nifti1Image(log_p_values, img.get_affine()),
              mean_img, title='Thresholded p-values', annotate=False,
              colorbar=True)



