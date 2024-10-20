import zarr

INT_FILENAME = "ext_rgb_img_record.zarr"
EXT_FILENAME = "int_rgb_img_record.zarr"

int_record_frames = zarr.load(INT_FILENAME)
ext_record_frames = zarr.load(EXT_FILENAME)

for i in range(0, len(frames)):
    cv2.waitKey(1)
    cv2.imshow("ext", int_record_frames[i])
    cv2.imshow("int", ext_record_frames[i])

cv2.destroyAllWindows()
