"""
Classes for taking input in other formats

.. inheritance-diagram:: proteus.InputTranslators
   :parts: 1
"""
from proteus.EGeometry import *
from .Profiling import logEvent

class GF(object):
    """
    Read a file from the vessel database an set up in a domain.
    """
    def __init__(self,gfFile,
                 boundingBox=False,
                 thin_rtol=0.1,
                 sN_restrict=-1,
                 insertHoles=False):
        #set up flags for facets in polyfile
        self.facetTypes={}
        for ki,k in enumerate(['left','right','front','back','bottom','top','solid']):
            self.facetTypes[k] = ki+1 #mwf add 1
        #read entire file
        f = open(gfFile,'r')
        lines = f.readlines()
        f.close()
        #write geometry to a polyfile
        self.polyfile = gfFile
        pf = open(self.polyfile+'.poly','w')
        #
        #start parsing gf file line by line
        #
        #name and reference number
        nl=0#line number
        pf.write('#Title:'+lines[nl]);nl+=1
        if lines[nl][0].isdigit():
            pf.write('#Reference Number:'+lines[1])
            nl+=1
        #comments
        comment=lines[nl]
        while comment:
            if comment[0] == "\\":
                pf.write('#Comments: '+comment)
                nl+=1
                comment=lines[nl]
            else:
                comment=False
        #miscellaneous information
        line = lines[nl]
        information={}
        while line[0] != '*':
            if line[0] != '\\':
                k,v = line.split(':')
                if k in ['L','W']:
                    information[k] = float(v)
                elif k == 'N':
                    information[k] = int(v)
                else:
                    information[k] = v.strip()
                pf.write('#Information: '+k+' '+str(information[k])+'\n')
            nl+=1
            line=lines[nl]
        #shapes
        shapes={}
        vertexDict={} #store vertices in a dictionary indexed by the Cartesian coordinate tuple
        nV=0#vertex number
        while '**' not in line:
            nl+=1
            line = lines[nl];nl+=1
            name = line.strip()
            shapes[name]={'sections':[],
                          'nSections':int(lines[nl])};nl+=1
            #sections in shape
            for sN in range(shapes[name]['nSections']):
                location,nVertices = lines[nl].split(',');nl+=1
                nVertices  = int(nVertices)
                section={'location':float(location),
                         'nVertices':nVertices,
                         'vertexList':[]}
                vertexDictSec={}
                #points in section
                for vLine in lines[nl:nl+nVertices]:
                    vertex = vLine.split(',')
                    fvertex =  (float(location),
                               float(vertex[0]),
                               float(vertex[1]))
                    if fvertex not in vertexDictSec:
                        vertexDictSec[fvertex] = nV; nV+=1
                    section['vertexList'].append((vertexDictSec[fvertex],fvertex))
                #check for thin section
                if sN > 1:
                    max_t = section['vertexList'][0][1][1]
                    max_z = section['vertexList'][0][1][2]
                    min_t = max_t; min_z=max_z;
                    for p in section['vertexList'][1:]:
                        max_t = max(max_t,p[1][1])
                        max_z = max(max_z,p[1][2])
                        min_t = min(min_t,p[1][1])
                        min_z = min(min_z,p[1][2])
                    L = min((max_t - min_t),(max_z - min_z))
                else:
                    L = 1.0
                nl+=nVertices
                if (sN > 0 and
                    (float(location) - shapes[name]['sections'][-1]['location']) < thin_rtol*L):
                    shapes[name]['nSections']-=1
                    nV-=  len(vertexDictSec)
                    logEvent("***************eliminating thin section********************")
                else:
                    vertexDict.update(vertexDictSec)
                    shapes[name]['sections'].append(section)
            logEvent("************reflecting along transverse axis****************************")
#             shapes[name+"_reflection"] = {'sections':[],
#                                           'nSections':shapes[name]['nSections']}
#             for section_right in shapes[name]['sections']:
#                section_left={'location':section_right['location'],
#                              'nVertices':section_right['nVertices'],
#                              'vertexList':[]}
#                #loop over points
#                for p in section_right['vertexList'][::-1]:
#                   fvertex =  (p[1][0],
#                              -p[1][1],
#                              p[1][2])
#                   if not vertexDict.has_key(fvertex):
#                      vertexDict[fvertex] = nV; nV+=1
#                   section_left['vertexList'].append((vertexDict[fvertex],fvertex))
#                if section_left['vertexList'][-1] != section_right['vertexList'][0]:
#                   if section_left['vertexList'].count(section_left['vertexList'][-1]) == 1:
#                      section_left['nVertices']+=1
#                      section_left['vertexList'].append(section_right['vertexList'][0])
#                if section_left['vertexList'][0] != section_right['vertexList'][-1]:
#                   if section_left['vertexList'].count(section_left['vertexList'][-1]) == 1:
#                      section_right['nVertices']+=1
#                      section_right['vertexList'].append(section_left['vertexList'][0])
#                shapes[name+"_reflection"]['sections'].append(section_left)
            for section_right in shapes[name]['sections']:
                #loop over points
                for p in section_right['vertexList'][::-1]:
                    fvertex =  (p[1][0],
                                -p[1][1],
                                p[1][2])
                    if fvertex not in vertexDict:
                        vertexDict[fvertex] = nV; nV+=1
                        section_right['nVertices']+=1
                        section_right['vertexList'].append((vertexDict[fvertex],fvertex))
                    elif (vertexDict[fvertex],fvertex) != section_right['vertexList'][-1]:
                        section_right['vertexList'].append((vertexDict[fvertex],fvertex))
                if section_right['vertexList'][0] != section_right['vertexList'][-1]:
                    section_right['vertexList'].append(section_right['vertexList'][0])
            vertexList = list(vertexDict.keys())
            for v,vN in vertexDict.items():
                vertexList[vN] = v
            #shell thickness
            if "PROP" not in lines[nl]:
                shapes[name]['shellThickness'] = [float(t) for t in lines[nl].split(',')]; nl+=1
            #property table
            if "PROP" in lines[nl]:
                shapes[name]['nProperties'] = int(lines['nl'].split(',')[1])
                shapes[name]['properties']=[]
                for nProp in range(shape[name]['nProperties']):
                    nl+=nProp
                    h,p = lines[nl].split()
                    shapes[name]['properties'].append((float(h),p))
            line = lines[nl]
        logEvent("Skipping components, parts, and critical points for now")
        #Components
        ##Parts
        #Critical Points
        #
        #now build poly file
        #
        #facets
        facets=[]#facets stored as sorted vertex number tuples to facilitate eliminating duplicate facets
        facetFlags=[]
        def appendEndFacets(pList,fList,ftList):
            if pList.count(pList[-1]) == 2:
                fList.append([p[0] for p in pList[:-1]])
                ftList.append(self.facetTypes['solid'])
            else:
                fList.append([p[0] for p in pList])
                ftList.append(self.facetTypes['solid'])
        for shape in list(shapes.values()):
            #front and back facet
            if sN_restrict == -1:
                appendEndFacets(shape['sections'][0]['vertexList'],facets,facetFlags)
                appendEndFacets(shape['sections'][shape['nSections']-1]['vertexList'],facets,facetFlags)
                sectionsToInclude = list(range(shape['nSections']-1))
            else:
                if shape == list(shapes.values())[0]:
                    appendEndFacets(shape['sections'][sN_restrict]['vertexList'],facets,facetFlags)
                    appendEndFacets(shape['sections'][sN_restrict+1]['vertexList'],facets,facetFlags)
                    sectionsToInclude = [sN_restrict]
                else:
                    break
            for sN in sectionsToInclude:
                pList_sN = shape['sections'][sN]['vertexList']
                pList_sNp1 = shape['sections'][sN+1]['vertexList']
                #check for loops
                points_sN=[]
                points_sNp1=[]
                points_sN = [p[0] for p in pList_sN]
                points_sNp1 = [p[0] for p in pList_sNp1]
                nLoops_sN=1
                for pN in points_sN[1:-1]:
                    if points_sN.count(pN) > 1:
                        nLoops_sN +=0.25
                nLoops_sNp1=1
                for pN in points_sNp1[1:-1]:
                    if points_sNp1.count(pN) > 1:
                        nLoops_sNp1 +=0.25
                if nLoops_sN > nLoops_sNp1:
                    logEvent("***********non-matching loops************ "+repr(nLoops_sN)+" "+repr(nLoops_sNp1))
                    #split points_sN into two loops
                    loop_1_sN=[points_sN.pop(0)]
                    loop_2_sN=[]
                    for p in points_sN[1:]:
                        if points_sN.count(p) == 2:
                            loop_1_sN.append(p)
                            I1 = points_sN.index(p)
                            I2 = I1+points_sN[I1+1:].index(p) + 1
                            loop_1_sN.extend(points_sN[I2+1:])
                            loop_2_sN  = points_sN[I1+1:I2]
                            break
                        else:
                            loop_1_sN.append(p)
                    min_edge_loop_1_sNp1=1.0e6
                    max_edge_loop_1_sNp1=0.0
                    for p_sNp1 in points_sNp1:
                        min_d=1.0e6;
                        for pN in loop_1_sN:
                            d = enorm(EVec(vertexList[p_sNp1][0]-vertexList[pN][0],
                                           vertexList[p_sNp1][1]-vertexList[pN][1],
                                           vertexList[p_sNp1][2]-vertexList[pN][2]))
                            min_d = min(d,min_d)
                        min_edge_loop_1_sNp1 = min(min_d,min_edge_loop_1_sNp1)
                        max_edge_loop_1_sNp1 = max(min_d,max_edge_loop_1_sNp1)
                    min_edge_loop_2_sNp1=1.0e6
                    max_edge_loop_2_sNp1=0.0
                    for p_sNp1 in points_sNp1:
                        min_d=1.0e6;
                        for pN in loop_2_sN:
                            d = enorm(EVec(vertexList[p_sNp1][0]-vertexList[pN][0],
                                           vertexList[p_sNp1][1]-vertexList[pN][1],
                                           vertexList[p_sNp1][2]-vertexList[pN][2]))
                            min_d = min(d,min_d)
                        min_edge_loop_2_sNp1 = min(min_d,min_edge_loop_2_sNp1)
                        max_edge_loop_2_sNp1 = max(min_d,max_edge_loop_2_sNp1)
                    if min_edge_loop_2_sNp1 < min_edge_loop_1_sNp1:
                        if max_edge_loop_2_sNp1 < min_edge_loop_1_sNp1:
                            pList_sN = [(pN,vertexList[pN]) for pN in loop_2_sN]
                            print("cut loop")
                            facets.append(loop_1_sN[:-1])
                            facetFlags.append(self.facetTypes['solid'])
                    elif min_edge_loop_1_sNp1 < min_edge_loop_2_sNp1:
                        if max_edge_loop_1_sNp1 < min_edge_loop_2_sNp1:
                            pList_sN = [(pN,vertexList[pN]) for pN in loop_1_sN]
                            print("cut loop")
                            facets.append(loop_2_sN[:-1])
                            facetFlags.append(self.facetTypes['solid'])
                if nLoops_sNp1 > nLoops_sN:
                    logEvent("***********non-matching loops************ "+repr(nLoops_sN)+" "+repr(nLoops_sNp1))
                    #split points_sN into two loops
                    loop_1_sNp1=[points_sNp1.pop(0)]
                    loop_2_sNp1=[]
                    for p in points_sNp1[1:]:
                        if points_sNp1.count(p) == 2:
                            loop_1_sNp1.append(p)
                            I1 = points_sNp1.index(p)
                            I2 = I1+points_sNp1[I1+1:].index(p) + 1
                            loop_1_sNp1.extend(points_sNp1[I2+1:])
                            loop_2_sNp1  = points_sNp1[I1+1:I2]
                            break
                        else:
                            loop_1_sNp1.append(p)
                    min_edge_loop_1_sN=1.0e6
                    max_edge_loop_1_sN=0.0
                    for p_sN in points_sN:
                        min_d=1.0e6;
                        for pN in loop_1_sNp1:
                            d = enorm(EVec(vertexList[p_sN][0]-vertexList[pN][0],
                                           vertexList[p_sN][1]-vertexList[pN][1],
                                           vertexList[p_sN][2]-vertexList[pN][2]))
                            min_d = min(d,min_d)
                        min_edge_loop_1_sN = min(min_d,min_edge_loop_1_sN)
                        max_edge_loop_1_sN = max(min_d,max_edge_loop_1_sN)
                    min_edge_loop_2_sN=1.0e6
                    max_edge_loop_2_sN=0.0
                    for p_sN in points_sN:
                        min_d=1.0e6;
                        for pN in loop_2_sNp1:
                            d = enorm(EVec(vertexList[p_sN][0]-vertexList[pN][0],
                                           vertexList[p_sN][1]-vertexList[pN][1],
                                           vertexList[p_sN][2]-vertexList[pN][2]))
                            min_d = min(d,min_d)
                        min_edge_loop_2_sN = min(min_d,min_edge_loop_2_sN)
                        max_edge_loop_2_sN = max(min_d,max_edge_loop_2_sN)
                    if min_edge_loop_2_sN < min_edge_loop_1_sN:
                        if max_edge_loop_2_sN < min_edge_loop_1_sN:
                            pList_sNp1 = [(pN,vertexList[pN]) for pN in loop_2_sNp1]
                            print("cut loop")
                            facets.append(loop_1_sNp1[:-1])
                            facetFlags.append(self.facetTypes['solid'])
                    elif min_edge_loop_1_sN < min_edge_loop_2_sN:
                        if max_edge_loop_1_sN < min_edge_loop_2_sN:
                            pList_sNp1 = [(pN,vertexList[pN]) for pN in loop_1_sNp1]
                            print("cut loop")
                            facets.append(loop_2_sNp1[:-1])
                            facetFlags.append(self.facetTypes['solid'])
                if len(pList_sN) > len(pList_sN):
                    pLong=pList_sN
                    pShort=pList_sNp1
                else:
                    pShort=pList_sN
                    pLong=pList_sNp1
                print(pLong)
                print(pShort)
                #assume first vertex in lists should be connected (could do nearest neighbor search)
                pNL=0
                pN=0
#                eLS=EVec(pLong[pNL][1][0]-pShort[pN][1][0],
#                         pLong[pNL][1][1]-pShort[pN][1][1],
#                         pLong[pNL][1][2]-pShort[pN][1][2])
#                eLSnorm_0 = enorm(eLS)
#                pNLtmp=1
#                #try search for closest nodes
#                while (pNLtmp < (len(pLong)-1)):
#                    eLSnorm=enorm(EVec(pLong[pNLtmp][1][0]-pShort[pN][1][0],
#                                       pLong[pNLtmp][1][1]-pShort[pN][1][1],
#                                       pLong[pNLtmp][1][2]-pShort[pN][1][2]))
#                    if eLSnorm < eLSnorm_0:
#                        eLSnorm_0 = eLSnorm
#                        pNL=pNLtmp
#                    pNLtmp += 1
                while (pN < (len(pShort)-1) and pNL < (len(pLong)-1)):
                    eLS=EVec(pLong[pNL][1][0]-pShort[pN+1][1][0],
                             pLong[pNL][1][1]-pShort[pN+1][1][1],
                             pLong[pNL][1][2]-pShort[pN+1][1][2])
                    eSL =EVec(pShort[pN][1][0]-pLong[pNL+1][1][0],
                              pShort[pN][1][1]-pLong[pNL+1][1][1],
                              pShort[pN][1][2]-pLong[pNL+1][1][2])
                    if enorm(eLS) < enorm(eSL):
                        facets.append([pLong[pNL][0],
                                       pShort[pN+1][0],
                                       pShort[pN][0]])
                        facets[-1].sort()
                        facetFlags.append(self.facetTypes['solid'])
                        pN+=1
                    else:
                        facets.append([pShort[pN][0],
                                       pLong[pNL+1][0],
                                       pLong[pNL][0]])
                        facets[-1].sort()
                        facetFlags.append(self.facetTypes['solid'])
                        pNL+=1
                while pN < (len(pShort)-1):
                    facets.append([pShort[pN][0],
                                   pShort[pN+1][0],
                                   pLong[pNL][0]])
                    facets[-1].sort()
                    facetFlags.append(self.facetTypes['solid'])
                    pN+=1
                while pNL < (len(pLong)-1):
                    facets.append([pLong[pNL+1][0],
                                   pLong[pNL][0],
                                   pShort[pN][0]])
                    facets[-1].sort()
                    facetFlags.append(self.facetTypes['solid'])
                    pNL+=1
                #assert(pLong[pNL][0] in facets[-1] and pShort[pN][0] in facets[-1])
                print(pLong[0],pLong[-1])
                print(pShort[0],pShort[-1])
                assert(pLong[0][0] == pLong[-1][0] and pShort[0][0] == pShort[-1][0])
                assert(pLong[0][0] in facets[-1] and pShort[0][0] in facets[-1])
#                assert(pNL == len(pLong)-1 and pN == len(pShort)-1)
#                print facets[-1]
#                print pLong[pNL],pShort[pN]
                #if only one section is a closed loop patch triangular hole
                if pShort.count(pShort[-1]) == 2 and pLong.count(pLong[-1]) == 1:
                    facets.append([pShort[-1][0],
                                   pLong[-1][0],
                                   pLong[0][0]])
                    facets[-1].sort()
                    facetFlags.append(self.facetTypes['solid'])
                if pShort.count(pShort[-1]) == 1 and pLong.count(pLong[-1]) == 2:
                    facets.append([pShort[-1][0],
                                   pLong[-1][0],
                                   pShort[0][0]])
                    facets[-1].sort()
                    facetFlags.append(self.facetTypes['solid'])
        #eliminate duplicate facets
        for f in facets:
            if facets.count(f) > 1:
                for i in range(facets.count(f)):
                    logEvent("***********deleting repeated facet************ "+repr(f))
                    I = facets.index(f)
                    del facets[I]
                #mwf delete facetFlag too?
                    del facetFlags[I]
        #find bounding box
        max_x_global = vertexList[0][0]
        max_y_global = vertexList[0][1]
        max_z_global = vertexList[0][2]
        min_x_global = vertexList[0][0]
        min_y_global = vertexList[0][1]
        min_z_global = vertexList[0][2]
        for p in vertexList:
            max_x_global = max(max_x_global,p[0])
            max_y_global = max(max_y_global,p[1])
            max_z_global = max(max_z_global,p[2])
            min_x_global = min(min_x_global,p[0])
            min_y_global = min(min_y_global,p[1])
            min_z_global = min(min_z_global,p[2])
        self.Lx = max_x_global-min_x_global
        self.Ly = max_y_global-min_y_global
        self.Lz = max_z_global-min_z_global
        self.Mx = 0.5*(max_x_global+min_x_global)
        self.My = 0.5*(max_y_global+min_y_global)
        self.Mz = 0.5*(max_z_global+min_z_global)
        if boundingBox:
            logEvent("Solid Bounding Box %12.5e %12.5e %12.5e" % (self.Lx,self.Ly,self.Lz))
            def xyz(i,j,k):
                return (self.Mx+i*self.Lx,
                        self.My+j*self.Ly,
                        self.Mz+k*self.Lz)
            #now try to include a midpoint in each face to make bc's easier to set?
            for i in [-1,0,1]:
                for j in [-1,0,1]:
                    for k in [-1,0,1]:
                        if not (i == 0 and j == 0 and k == 0):
                            vertexDict[xyz(i,j,k)]=nV;nV+=1

            #
            for j in [-1,0]:
                for k in [-1,0]:
                    facets.append([vertexDict[xyz(-1,j,k)],
                                   vertexDict[xyz(-1,j+1,k)],
                                   vertexDict[xyz(-1,j+1,k+1)],
                                   vertexDict[xyz(-1,j,k+1)]])
                    facetFlags.append(self.facetTypes['left'])

            #
            for j in [-1,0]:
                for k in [-1,0]:
                    facets.append([vertexDict[xyz(1,j,k)],
                                   vertexDict[xyz(1,j+1,k)],
                                   vertexDict[xyz(1,j+1,k+1)],
                                   vertexDict[xyz(1,j,k+1)]])
                    facetFlags.append(self.facetTypes['right'])
            #
            for i in [-1,0]:
                for k in [-1,0]:
                    facets.append([vertexDict[xyz(i,-1,k)],
                                   vertexDict[xyz(i+1,-1,k)],
                                   vertexDict[xyz(i+1,-1,k+1)],
                                   vertexDict[xyz(i,-1,k+1)]])
                    facetFlags.append(self.facetTypes['front'])

            #
            for i in [-1,0]:
                for k in [-1,0]:
                    facets.append([vertexDict[xyz(i,1,k)],
                                   vertexDict[xyz(i+1,1,k)],
                                   vertexDict[xyz(i+1,1,k+1)],
                                   vertexDict[xyz(i,1,k+1)]])
                    facetFlags.append(self.facetTypes['back'])
            #
            for i in [-1,0]:
                for j in [-1,0]:
                    facets.append([vertexDict[xyz(i,j,-1)],
                                   vertexDict[xyz(i+1,j,-1)],
                                   vertexDict[xyz(i+1,j+1,-1)],
                                   vertexDict[xyz(i,j+1,-1)]])
                    facetFlags.append(self.facetTypes['bottom'])

            #
            for i in [-1,0]:
                for j in [-1,0]:
                    facets.append([vertexDict[xyz(i,j,1)],
                                   vertexDict[xyz(i+1,j,1)],
                                   vertexDict[xyz(i+1,j+1,1)],
                                   vertexDict[xyz(i,j+1,1)]])
                    facetFlags.append(self.facetTypes['top'])
        pf.write('#vertices\n')
        if sN_restrict == -1:
            pN0=0
            pf.write(str(len(vertexDict))+' 3 0 1\n')
            vertexList = [(pN+1,p[0],p[1],p[2],0) for p,pN in vertexDict.items()]
            vertexList.sort()
            for p in vertexList:
                pf.write('%d %e %e %e %d\n' % p)
        else:
            shapeList=list(shapes.values())
            vertexSet=set()
            for sN in sectionsToInclude:
                vertexSet |= set(shapeList[-1]['sections'][sN]['vertexList'])
            pN0 = shapeList[-1]['sections'][int(sectionsToInclude[0])]['vertexList'][0][0]
            pf.write(str(len(vertexSet))+' 3 0 1\n')
            for v in vertexSet:
                pf.write('%d %e %e %e %d\n' % (v[0]-pN0,v[1][0],v[1][1],v[1][2],0))
        pf.write('#facets\n')
        pf.write(str(len(facets))+' 1\n')
        nf=0
        for f,fF in zip(facets,facetFlags):
            nf+=1
            pf.write('1 0 %d #%d\n' % (fF,nf))
            pf.write(str(len(f)))
            for vN in f: pf.write(' '+str(vN+1-pN0))
            pf.write('\n')
        #mwf hack
        if not insertHoles:
            pf.write('#holes\n0\n#regions\n0\n')
        else:
            pf.write('#holes\n')
            pf.write('%d\n' % (1,))
            pf.write('%d %12.5e  %12.5e %12.5e %d\n' % (1,
                                                        self.Mx,
                                                        self.My,
                                                        self.Mz,
                                                        0+1))
            pf.write('#regions\n0\n')
        pf.close()


class Ipars(object):
    """
    Extract as much as possible from and IPARS input file
    """
    def __init__(self,iparsFile,layered=False):
        #read file into a dictionary
        f = open(iparsFile,'r')
        lines = f.readlines()
        values={}
        for ln,line in enumerate(lines):
            if "$" in line:
                continue
            if '=' in line:
                words = line.split('=')
                if '(' in words[0]:
                    key = words[0].split('(')[0].strip()
                else:
                    key = words[0].strip()
                values[key] = []
                if len(words) > 1:
                    if "\"" in words[1]:
                        values[key].append(words[1])
                    else:
                        for val in words[1].split():
                            try:
                                if "*" in val:
                                    n,v = val.split('*')
                                    for i in range(int(n)):
                                        values[key].append(eval(v))
                                else:
                                    values[key].append(eval(val))
                            except:
                                values[key].append(val)
                for vline in lines[ln+1:]:
                    if ("=" in vline or "\n" == vline or "$" in vline):
                        break
                    if "\"" in vline:
                        values[key].append(vline)
                    else:
                        vals = vline.split()
                        for val in vals:
                            try:
                                if "*" in val:
                                    n,v = val.split('*')
                                    for i in range(int(n)):
                                        values[key].append(eval(v))
                                else:
                                    values[key].append(eval(val))
                            except:
                                values[key].append(val)
        for key,value in values.items():
            logEvent(key+":"+str(value))
        #
        #MESH. Write a poly file matching the grid
        #
        pf = open(iparsFile+'.poly','w')
        self.polyfile=iparsFile
        x=values['XYZ111'][0]
        y=values['XYZ111'][1]
        z=values['XYZ111'][2]
        dxList = [0.0]+values['DX']
        dyList = [0.0]+values['DY']
        dzList = [0.0]+values['DZ']
        Lx = sum(dxList)
        Ly = sum(dyList)
        Lz = sum(dzList)
        nnx = len(dxList)
        nny = len(dyList)
        nnz = len(dzList)
        nVertices = nnx*nny*nnz
        #vertices
        pf.write('#vertices\n')
        pf.write('%d 3 0 1\n' % nVertices)
        vN=0
        vertices=[]
        vertexNumber={}
        self.boundaryFlags={'left':1,'right':2,'front':3,'back':4,'bottom':5,'top':6,'interior':0}
        boundaryFlags = self.boundaryFlags
        def getBoundaryFlag(i,j,k):
            if k==0:
                return boundaryFlags['bottom']
            if k==nnz-1:
                return boundaryFlags['top']
            if j==0:
                return boundaryFlags['front']
            if j==nny-1:
                return boundaryFlags['back']
            if i==0:
                return boundaryFlags['left']
            if i==nnx-1:
                return boundaryFlags['right']
            else:
                return boundaryFlags['interior']
        for i,dx in enumerate(dxList):
            x+=dx
            y=0.0
            z=0.0
            for j,dy in enumerate(dyList):
                y+=dy
                z=0.0
                for k,dz in enumerate(dzList):
                    z+=dz
                    vertexNumber[(i,j,k)]=vN
                    vertices.append((x,y,z))
                    pf.write("%d %12.5e %12.5e %12.5e %d\n" % (vN+1,x*0.3048,y*0.3048,z*0.3048,getBoundaryFlag(i,j,k)) )#cek hack,write out in meters
                    vN+=1
        #facets
        pf.write('#facets\n')
        facets=set()
        for i in range(nnx-1):
            for j in range(nny-1):
                for k in range(nnz-1):
                    facets|=set([(vertexNumber[(i,j,k)],
                                  vertexNumber[(i,j,k+1)],
                                  vertexNumber[(i,j+1,k+1)],
                                  vertexNumber[(i,j+1,k)],
                                  getBoundaryFlag(i,-1,-1)),
                                 (vertexNumber[(i+1,j,k)],
                                  vertexNumber[(i+1,j,k+1)],
                                  vertexNumber[(i+1,j+1,k+1)],
                                  vertexNumber[(i+1,j+1,k)],
                                  getBoundaryFlag(i+1,-1,-1)),
                                 (vertexNumber[(i,j,k)],
                                  vertexNumber[(i,j,k+1)],
                                  vertexNumber[(i+1,j,k+1)],
                                  vertexNumber[(i+1,j,k)],
                                  getBoundaryFlag(-1,j,-1)),
                                 (vertexNumber[(i,j+1,k)],
                                  vertexNumber[(i,j+1,k+1)],
                                  vertexNumber[(i+1,j+1,k+1)],
                                  vertexNumber[(i+1,j+1,k)],
                                  getBoundaryFlag(-1,j+1,-1)),
                                 (vertexNumber[(i,j,k)],
                                  vertexNumber[(i,j+1,k)],
                                  vertexNumber[(i+1,j+1,k)],
                                  vertexNumber[(i+1,j,k)],
                                  getBoundaryFlag(-1,-1,k)),
                                 (vertexNumber[(i,j,k+1)],
                                  vertexNumber[(i,j+1,k+1)],
                                  vertexNumber[(i+1,j+1,k+1)],
                                  vertexNumber[(i+1,j,k+1)],
                                  getBoundaryFlag(-1,-1,k+1))])
        pf.write('%d 1 \n' % (len(facets),))
        for f in facets:
            pf.write('1 0 %d\n' % f[-1])
            pf.write('4 %d %d %d %d\n' % (f[0]+1,f[1]+1,f[2]+1,f[3]+1,))
        pf.write('#holes\n')
        pf.write('0 \n')
        pf.write('#regions\n')
        nCells=(nnx-1)*(nny-1)*(nnz-1)
        if layered:
            pf.write('%d \n' % (nCells))
            rN=1
            for i in range(nnx-1):
                for j in range(nny-1):
                    for k in range(nnz-1):
                        p1 = vertices[vertexNumber[(i,j,k)]]
                        p2 = vertices[vertexNumber[(i+1,j+1,k+1)]]
                        pi = (0.5*(p1[0]+p2[0]),0.5*(p1[1]+p2[1]),0.5*(p1[2]+p2[2]))
                        pf.write('%d %12.5e %12.5e %12.5e %d\n' % (rN,pi[0]*0.3048,pi[1]*0.3048,pi[2]*0.3048,k))
                        rN+=1
        else:
            pf.write('1 \n')
            pf.write('1 %12.5e %12.5e %12.5e 0 0.0\n' % (1.0,1.0,1.0))
        pf.close()
        self.values=values
        self.L=(Lx*0.3048,Ly*0.3048,Lz*0.3048)


class ADH_metfile(object):
    """
    read an ADH met file for boundary conditions etc
    """
    from proteus.cSubsurfaceTransportCoefficients import piecewiseLinearTableLookup
    # staticmethod: Cython 3 functions bind self like Python functions (C47)
    piecewiseLinearTableLookup = staticmethod(piecewiseLinearTableLookup)
    allowed_time_units = ['day','hour','sec']
    def __init__(self,fileprefix,directory='.'):
        self.fileprefix=fileprefix
        self.directory = directory

        import os,numpy

        filename = os.path.join(directory,fileprefix+'.met')
        assert os.path.exists(filename), "%s not found" % filename

        fh = open(filename,'r')
        self.full_file = fh.readlines()
        fh.close()

        #
        latitude = float(self.full_file[2].split()[0]); longitude=float(self.full_file[2].split()[1]); zone=float(self.full_file[2].split()[2]);

        self.npoints = len(self.full_file[5:])
        self.data = {}
        self.data['latitude']=latitude; self.data['longitude']=longitude; self.data['zone']=zone
        #default things to look at
        #Time is in separate day,hour,min
        self.entries = {'day':0,'hour':1,'min':2}
        for  i,label in enumerate(self.full_file[3].split()[2:]):
            self.entries[label] = i+3
        for entry in list(self.entries.keys()):
            self.data[entry] = numpy.zeros((self.npoints),'d')

        for i,line in enumerate(self.full_file[5:]):
            for entry,column in self.entries.items():
                self.data[entry][i] = float(line.split()[column])

        #generate a time entry out of hours days minutes
        self.time_unit = 'day' #hour, sec
        self.data['time'] = numpy.zeros((self.npoints),'d')
        self.data['time'] = self.data['day']+self.data['hour']/24.0 + self.data['min']/(24.*60.)
    def getValue(self,entry,t):
        """
        lookup value of entry at time t
        """
        if entry not in list(self.data.keys()):
            print("ADH_metfile entry= %s not found " % entry)
            return None
        if entry in ['latitude','longitude','zone']:
            return self.data[entry]
        index = 0
        interp,dinterp,index = self.piecewiseLinearTableLookup(t,self.data['time'],self.data[entry],index)
        return interp