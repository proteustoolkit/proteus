import numpy as np


def InitializeXdmf(filename="beam"):
    f = open(filename + ".xmf", 'w')
    f.write('<?xml version="1.0" ?>' + '\n'
            + '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>' + '\n'
            + '<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">' + '\n'
            + '\t<Domain> \n'
            + '\t\t<Grid CollectionType="Temporal" GridType="Collection" Name="Mesh Spatial_Domain"> \n'
            + '\t\t</Grid> \n'
            + '\t</Domain> \n'
            + '</Xdmf>')
    f.close()


def AddTimestep(Beam_x,
                Beam_y,
                Beam_z,
                nBeams,
                filename="beam",
                t=0.0):
    f = open(filename + ".xmf", 'r')
    data_list = f.readlines()
    f.close()
    print(data_list)
    del data_list[-1]
    del data_list[-1]
    del data_list[-1]

    f = open(filename + ".xmf", 'w')
    f.writelines(data_list)
    f.write('\t\t\t<Grid GridType="Uniform">\n'
            + '\t\t\t\t<Time Value="' + repr(t) + '" />\n'
            + '\t\t\t\t<Topology TopologyType="Polyline" Dimensions="' + repr(nBeams) + '">\n'
            + '\t\t\t\t\t<DataItem Dimensions="' + repr(nBeams) +' ' + repr(Beam_x[0].size) + '" NumberType="Int" Precision="8" Format="XML">\n')
    count = 0
    for i in range(len(Beam_x)):
        for j in range(Beam_x[i].size):
            f.write(repr(count) + ' ')
            count += 1
        f.write('\n')

    f.write('\t\t\t\t\t</DataItem>\n'
            + '\t\t\t\t</Topology>\n'
            + '\t\t\t\t<Geometry GeometryType="XYZ">\n'
            + '\t\t\t\t\t<DataItem Dimensions="' + repr(len(Beam_x) * Beam_x[0].size) +' 3" NumberType="Float" Precision="4" Format="XML">\n')
    for i in range(len(Beam_x)):
        for j in range(Beam_x[i].size):
            sp = repr(Beam_x[i][j]) + ' ' + repr(Beam_y[i][j]) + ' ' + repr(Beam_z[i][j]) + '\n'
            f.write(sp)
    f.write('\t\t\t\t\t</DataItem>\n'
            + '\t\t\t\t</Geometry>\n'
            + '\t\t\t</Grid>\n'
            + '\t\t</Grid>\n'
            + '\t</Domain>\n'
            + '</Xdmf>\n')
    f.close()


def Archive_parallel(Beam_x,
                     Beam_y,
                     Beam_z,
                     nBeams,
                     filename="beam",
                     tList=[0.0]):
    f = open(filename + ".xmf", 'w')
    f.write('<?xml version="1.0" ?>' + '\n'
            + '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>' + '\n'
            + '<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">' + '\n'
            + '\t<Domain> \n'
            + '\t\t<Grid CollectionType="Temporal" GridType="Collection" Name="Mesh Spatial_Domain"> \n')
    for k in range(len(tList)):
        f.write('\t\t\t<Grid GridType="Uniform">\n'
                + '\t\t\t\t<Time Value="' + repr(tList[k]) + '" />\n'
                + '\t\t\t\t<Topology TopologyType="Polyline" Dimensions="' + repr(nBeams) + '">\n'
                + '\t\t\t\t\t<DataItem Dimensions="' + repr(nBeams) +' ' + repr(Beam_x[k][0].size) + '" NumberType="Int" Precision="8" Format="XML">\n')
        count = 0
        for i in range(len(Beam_x[k])):
            for j in range(Beam_x[k][i].size):
                f.write(repr(count) + ' ')
                count += 1
            f.write('\n')

        f.write('<?xml version="1.0" ?>' + '\n'
                + '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>' + '\n'
                + '<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">' + '\n'
                + '\t<Domain> \n'
                + '\t\t<Grid CollectionType="Temporal" GridType="Collection" Name="Mesh Spatial_Domain"> \n'
                '\t\t\t\t\t</DataItem>\n'
                + '\t\t\t\t</Topology>\n'
                + '\t\t\t\t<Geometry GeometryType="XYZ">\n'
                + '\t\t\t\t\t<DataItem Dimensions="' + repr(len(Beam_x[k]) * Beam_x[k][0].size) +' 3" NumberType="Float" Precision="4" Format="XML">\n')
        for i in range(len(Beam_x[k])):
            for j in range(Beam_x[k][i].size):
                sp = repr(Beam_x[k][i][j]) + ' ' + repr(Beam_y[k][i][j]) + ' ' + repr(Beam_z[k][i][j]) + '\n'
                f.write(sp)
        f.write('\t\t\t\t\t</DataItem>\n'
                + '\t\t\t\t</Geometry>\n'
                + '\t\t\t</Grid>\n')
    f.write('\t\t</Grid>\n'
            + '\t</Domain>\n'
            + '</Xdmf>\n')

    f.close()


def Archive_time_step(Beam_x,
                      Beam_y,
                      Beam_z,
                      nBeams,
                      filename="beam",
                      t=0.0,
                      tStep=0):

    f = open(filename + repr(tStep)+".xmf", 'w')
    # f.writelines(data_list)
    f.write('<?xml version="1.0" ?>' + '\n'
            + '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>' + '\n'
            + '<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">' + '\n'
            + '\t<Domain> \n'
            + '\t\t<Grid CollectionType="Temporal" GridType="Collection" Name="Mesh Spatial_Domain"> \n'
            + '\t\t\t<Grid GridType="Uniform">\n'
            + '\t\t\t\t<Time Value="' + repr(t) + '" />\n'
            + '\t\t\t\t<Topology TopologyType="Polyline" Dimensions="' + repr(nBeams) + '">\n'
            + '\t\t\t\t\t<DataItem Dimensions="' + repr(nBeams) +' ' + repr(Beam_x[0].size) + '" NumberType="Int" Precision="8" Format="XML">\n')
    count = 0
    for i in range(len(Beam_x)):
        for j in range(Beam_x[i].size):
            f.write(repr(count) + ' ')
            count += 1
        f.write('\n')

    f.write('\t\t\t\t\t</DataItem>\n'
            + '\t\t\t\t</Topology>\n'
            + '\t\t\t\t<Geometry GeometryType="XYZ">\n'
            + '\t\t\t\t\t<DataItem Dimensions="' + repr(len(Beam_x) * Beam_x[0].size) +' 3" NumberType="Float" Precision="4" Format="XML">\n')
    for i in range(len(Beam_x)):
        for j in range(Beam_x[i].size):
            sp = repr(Beam_x[i][j]) + ' ' + repr(Beam_y[i][j]) + ' ' + repr(Beam_z[i][j]) + '\n'
            f.write(sp)
    f.write('\t\t\t\t\t</DataItem>\n'
            + '\t\t\t\t</Geometry>\n'
            + '\t\t\t</Grid>\n'
            + '\t\t</Grid>\n'
            + '\t</Domain>\n'
            + '</Xdmf>\n')
    f.close()
