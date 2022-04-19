import os 


def getTileFileFromDimensions(tiles_path, dim1, dim2):
    tile_file='tile_'+str(dim1)+'x'+str(dim2)
    return os.path.join(tiles_path, tile_file)

