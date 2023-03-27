import dicom2nifti
import nibabel as nib
import os
import numpy as np
from PIL import Image
import zipfile
import h5py
import json


def adequate_ircad():

    dir_data = "/media/juliosilva/Seagate Portable Drive/LIVIA/Datasets/UniversalOrganSegmentation/06_3D-IRCADb/"

    classes_dict = {'artery': 1, 'bone': 2, 'leftkidney': 3, 'leftlung': 4, 'liver': 5, 'liverkyst': 6, 'livertumor': 7,
                    'portalvein': 8, 'rightkidney': 9, 'rightlung': 10, 'skin': 11, 'spleen': 12, 'venoussystem': 13,
                    'gall': 14, 'stomach': 15, 'pancreas': 16, 'lungs': 17, }
    with open(dir_data + 'labels_indexes.txt', "w") as fp:
        json.dump(classes_dict, fp)

    if not os.path.exists(dir_data + 'img/'):
        os.mkdir(dir_data + 'img/')

    files = os.listdir(dir_data + 'data/')
    for iFile in files:
        print(iFile)

        # Unzip images and labels
        with zipfile.ZipFile(dir_data + 'data/' + iFile + '/LABELLED_DICOM.zip', 'r') as zip_ref:
            zip_ref.extractall(dir_data + 'data/' + iFile)
        with zipfile.ZipFile(dir_data + 'data/' + iFile + '/MASKS_DICOM.zip', 'r') as zip_ref:
            zip_ref.extractall(dir_data + 'data/' + iFile)

        # Dicom image to .nii and save
        if not os.path.isfile(dir_data + 'img/' + iFile.replace('.', '_') + '.nii.gz'):
            dicom2nifti.convert_dicom.dicom_series_to_nifti(dir_data + 'data/' + iFile + '/LABELLED_DICOM/',
                                                            dir_data + 'img/' + iFile.replace('.','_') + '.nii.gz')

        # Convert masks to .nii volumes
        if not os.path.isfile(dir_data + 'data/' + iFile + '/MASKS_NII/'):
            os.mkdir(dir_data + 'data/' + iFile + '/MASKS_NII/')

            masks_files = os.listdir(dir_data + 'data/' + iFile + '/MASKS_DICOM/')
            for iMask in masks_files:
                dicom2nifti.convert_dicom.dicom_series_to_nifti(dir_data + 'data/' + iFile + '/MASKS_DICOM/' + iMask,
                                                                dir_data + 'data/' + iFile + '/MASKS_NII/' + iMask + '.nii.gz')


def adequate_ircad_2():

    dir_data = "/media/juliosilva/Seagate Portable Drive/LIVIA/Datasets/UniversalOrganSegmentation/06_3D-IRCADb/"

    all_annotations = []
    files = os.listdir(dir_data + 'data/')
    for iFile in files:
        print(iFile)

        # Merge masks into one unique file
        masks_files_nii = os.listdir(dir_data + 'data/' + iFile + '/MASKS_NII/')
        for iMask in masks_files_nii:

            if ('tumor' not in iMask.split('.')[0]) and ('portalvein' not in iMask.split('.')[0]):
                all_annotations.append(iMask.split('.')[0])
            elif 'livertumor' in iMask.split('.')[0]:
                all_annotations.append('livertumor')
            elif iMask.split('.')[0] == 'tumor':
                all_annotations.append('tumor')
            elif 'portalvein' in iMask.split('.')[0]:
                all_annotations.append('portalvein')

    unique_organs, counts = np.unique(all_annotations, return_counts=True)
    unique_organs, counts = list(unique_organs), list(counts)

    # Create dictionary by importance sorting
    idx = list(np.argsort(counts))
    idx.reverse()
    annotations_dict = {}
    c = 0
    for i in idx:
        c+=1
        annotations_dict[unique_organs[i]] = c


def adequate_ircad_3():

    dir_data = "/media/juliosilva/Seagate Portable Drive/LIVIA/Datasets/UniversalOrganSegmentation/06_3D-IRCADb/"

    if not os.path.exists(dir_data + 'post_label/'):
        os.mkdir(dir_data + 'post_label/')

    # Unique, different categories found in the dataset
    cats_ircad = {'livertumor': 1, 'skin': 2, 'bone': 3, 'liver': 4, 'portalvein': 5, 'artery': 6, 'venoussystem': 7,
                  'venacava': 8, 'gallbladder': 9, 'leftkidney': 10, 'rightkidney': 11, 'spleen': 12, 'rightlung': 13,
                  'leftlung': 14, 'leftsurrenalgland': 15, 'lungs': 16, 'pancreas': 17, 'stomach': 18,
                  'rightsurrenalgland': 19, 'heart': 20, 'biliarysystem': 21, 'bladder': 22, 'tumor': 23, 'uterus': 24,
                  'colon': 25, 'smallintestin': 26, 'livercyst': 27, 'kidneys': 28, 'surrenalgland': 29,
                  'liverkyst': 30, 'liverkyste': 31, 'metal': 32, 'metastasectomie': 33, 'Stones': 34}

    files = os.listdir(dir_data + 'data/')
    for iFile in files:
        print(iFile)

        # Merge masks into one unique file
        masks_files_nii = os.listdir(dir_data + 'data/' + iFile + '/MASKS_NII/')
        annotations = []
        cats = []
        for iMask in masks_files_nii:
            vol = nib.load(dir_data + 'data/' + iFile + '/MASKS_NII/' + iMask)
            vol_values = np.array(vol.dataobj) // 255

            annotations.append(vol_values)

            if ('tumor' not in iMask.split('.')[0]) and ('portalvein' not in iMask.split('.')[0]):
                cats.append(iMask.split('.')[0])
            elif 'livertumor' in iMask.split('.')[0]:
                cats.append('livertumor')
            elif iMask.split('.')[0] == 'tumor':
                cats.append('tumor')
            elif 'portalvein' in iMask.split('.')[0]:
                cats.append('portalvein')

        out = np.zeros([len(cats_ircad)] + list(annotations[0].shape))
        for i in np.arange(0, len(annotations)):
            out[cats_ircad[cats[5]], :, :, :] += annotations[i]
        out = np.expand_dims(out, axis=0)

        with h5py.File(dir_data + 'post_label/' + iFile.replace('.', '_') + '.h5', 'w') as f:
            f.create_dataset('post_label', data=out.astype(np.uint8), compression='gzip', compression_opts=9)
            f.close()

        #nifti_file = nib.Nifti1Image(np.int32(out[0, :, :, :]), affine=vol.affine)
        nib.save(vol, dir_data + 'label/' + iFile.replace('.', '_') + '.nii.gz')


def adequate_tcia():

    dir_data = "E:/LIVIA/Datasets/UniversalOrganSegmentation/02_TCIA_Pancreas-CT/volumes/"
    dir_out = "C:/Users/jusir/Documents/Research/Projects/UniversalModel/temp/"

    files = os.listdir(dir_data)

    for iFile in files:
        path_1 = os.listdir(dir_data + iFile)
        path_2 = os.listdir(dir_data + iFile + "/" + path_1[-1])
        path_volume = dir_data + iFile + "/" + path_1[-1] + "/" + path_2[-1]

        if not os.path.isfile(dir_out + iFile + '.nii.gz'):
            dicom2nifti.convert_dicom.dicom_series_to_nifti(path_volume, dir_out + iFile + '.nii.gz')


def adequate_lits():

    dir_in = "C:/Users/jusir/Desktop/lits_raw/segmentations/"
    dir_out = "C:/Users/jusir/Documents/Research/Projects/UniversalModel/temp/label/"

    files = os.listdir(dir_in)
    for iFile in files:
        print(iFile)

        id = iFile.split('-')[-1].split('.')[0]
        new_name = 'liver_' + id + '.nii'

        vol = nib.load(dir_in + iFile)
        nib.save(vol, dir_out + new_name + '.gz')


def adequate_chaos():

    dir_data = "E:/LIVIA/Datasets/UniversalOrganSegmentation/03_CHAOS/raw/"
    dir_out = "C:/Users/jusir/Documents/Research/Projects/UniversalModel/temp/"
    modalities = ["CT"]
    partitions = ["Train_Sets"]

    for iPartition in partitions:
        for iModality in modalities:

            if not os.path.exists(dir_out + iModality.lower()):
                os.mkdir(dir_out + iModality.lower())
                os.mkdir(dir_out + iModality.lower() + "/img/")
                os.mkdir(dir_out + iModality.lower() + "/label/")

            files = os.listdir(dir_data + iPartition + "/" + iModality + "/")

            for iFile in files:
                print(iFile)

                # Volume
                path_volume = dir_data + iPartition + "/" + iModality + "/" + iFile + "/DICOM_anon/"
                if not os.path.isfile(dir_out + iModality.lower() + "/img/" + iFile + "_image.nii.gz"):
                    dicom2nifti.convert_dicom.dicom_series_to_nifti(path_volume,
                                                                    dir_out + iModality.lower() + "/img/" + iFile + '_image.nii.gz')

                # Labels
                path_volume = dir_data + iPartition + "/" + iModality + "/" + iFile + "/Ground/"
                if not os.path.isfile(dir_out + iModality.lower() + "/label/" + iFile + "_segmentation.gz"):

                    annotations = os.listdir(path_volume)
                    vol_segm = []

                    for iSlide in annotations:
                        im = np.int32(np.asarray(Image.open(path_volume + iSlide)))
                        im = np.rot90(im, 3, (0, 1))
                        vol_segm.append(np.expand_dims(im, axis=-1))

                    vol_segm.reverse()
                    vol = np.concatenate(vol_segm, -1)
                    nifti_file = nib.Nifti1Image(vol, affine=None)
                    nib.save(nifti_file, dir_out + iModality.lower() + "/label/" + iFile + "_segmentation.nii.gz")


def adequate_kits():

    dir_data = "C:/Users/jusir/Documents/Research/Projects/UniversalModel/temp/kits21/"

    files = os.listdir(dir_data + "kits21/data/")
    for iFile in files:
        print(iFile, end='\n')

        if 'case' in iFile:
            id = iFile.split('_')[-1][1:]

            vol = nib.load(dir_data + "kits21/data/" + iFile + "/imaging.nii.gz")
            nib.save(vol, dir_data + "/img/" + "img" + id + ".nii.gz")

            label = nib.load(dir_data + "kits21/data/" + iFile + "/aggregated_MAJ_seg.nii.gz")
            nib.save(label, dir_data + "/label/" + "label" + id + ".nii.gz")