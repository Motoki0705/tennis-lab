"""Numerical checks for this offline reference experiment (CPU only)."""
import numpy as np
from scipy.special import erf

import fit_reference as fit


def test_paired_edges_measure_the_centre_not_the_inner_edge():
    offsets=np.arange(-48,48.01,.5);centre=2.25;width=7.5;sigma=.8
    profile=40+40*(erf((offsets-(centre-width/2))/sigma)-erf((offsets-(centre+width/2))/sigma))
    pair=fit.stripe_pairs(profile,offsets)[0]
    assert abs(pair[0]-centre)<.08
    assert abs(pair[2]-width)<.6
    assert abs(pair[0]-(centre-width/2))>3


def test_dark_shadow_is_not_a_bright_paint_stripe():
    offsets=np.arange(-48,48.01,.5)
    profile=100-40*(erf((offsets+3)/.8)-erf((offsets-3)/.8))
    assert fit.stripe_pairs(profile,offsets)==[]


def test_line_fit_recovers_known_projective_mapping_with_outliers():
    truth=np.array([[28.,33.,800.],[5.,-19.,500.],[.01,.02,1.]])
    initial=truth.copy();initial[:2,2]+=[5,-4]
    points=[];labels=[]
    rng=np.random.default_rng(93)
    for index,(a,b) in enumerate(fit.EDGES):
        xy=(1-np.linspace(.05,.95,24))[:,None]*fit.POINTS[a]+np.linspace(.05,.95,24)[:,None]*fit.POINTS[b]
        homogeneous=np.column_stack((xy,np.ones(len(xy))))@truth.T
        image=homogeneous[:,:2]/homogeneous[:,2,None]
        image+=rng.normal(0,.05,image.shape)
        image[:2]+=[20,-15]
        endpoints=np.column_stack((fit.POINTS[[a,b]],np.ones(2)))@truth.T
        endpoints=endpoints[:,:2]/endpoints[:,2,None];delta=endpoints[1]-endpoints[0]
        normal=np.array([-delta[1],delta[0]])/np.linalg.norm(delta)
        observations={'index':index,'status':'extracted','start':endpoints[0].tolist(),'end':endpoints[1].tolist(),
            'locations':24,'normal':normal.tolist(),'samples':[[*p,0,i,40,2,0] for i,p in enumerate(image)]}
        line=fit.fit_line(observations,minimum_frames=1)
        assert line['status']=='ok' and not set(line['inliers'])&{0,1}
        measured=image[line['inliers']]
        points.extend(measured);labels.extend([index]*len(measured))
    h,diagnostic=fit.fit_h(initial,np.array(points),np.array(labels))
    assert diagnostic['success'] and diagnostic['rank']==8
    truth_points=np.column_stack((fit.POINTS,np.ones(14)))@truth.T
    expected=truth_points[:,:2]/truth_points[:,2,None]
    assert np.max(np.linalg.norm(fit.project(h)-expected,axis=1))<.1
    assert not fit.valid_h(np.array([[1.,0,0],[0,1,0],[1,0,0]]))
