import jax
import jax.numpy as jnp

def generate_tensile_bar_indices(grip_width:            float,
                                 grip_length:           float,
                                 gage_width:            float,
                                 gage_length:           float,
                                 thickness:             float,
                                 fillet_major_diameter: float,
                                 fillet_minor_diameter: float,
                                 ideal_spacing:         float
                                 ) -> jnp.array:
    # just to get the coordinates straight, (0,0,0) is in the middle of the gage section, middle of the thickness
    # the y axis is pointing vertically along the tensile direction

    fillet_center_x = fillet_minor_diameter/2 + gage_width/2
    fillet_center_y = gage_length/2
    fillet_upper_corner_x = grip_width/2
    fillet_length = ((1 - (grip_width/2 - fillet_center_x)**2 / (fillet_minor_diameter/2)**2) * (fillet_major_diameter/2)**2)**0.5
    fillet_upper_corner_y = fillet_length + fillet_center_y

    # Lower Grip
    num_p_grip_x = int(grip_width/ideal_spacing)+1
    num_p_grip_y = int(grip_length/ideal_spacing)
    num_p_grip_z = int(thickness/ideal_spacing)+1
    section_ids = jnp.repeat(0,num_p_grip_x*num_p_grip_y*num_p_grip_z)
    widths = jnp.repeat(num_p_grip_x,num_p_grip_x*num_p_grip_y*num_p_grip_z)
    heights = jnp.repeat(num_p_grip_y,num_p_grip_x*num_p_grip_y*num_p_grip_z)
    thicknesses = jnp.repeat(num_p_grip_z,num_p_grip_x*num_p_grip_y*num_p_grip_z)
    x_ids = jnp.repeat(jnp.repeat(jnp.arange(0,num_p_grip_x),num_p_grip_y),num_p_grip_z)
    y_ids = jnp.tile(jnp.repeat(jnp.arange(0,num_p_grip_y),num_p_grip_z),num_p_grip_x)
    z_ids = jnp.tile(jnp.tile(jnp.arange(0,num_p_grip_z),num_p_grip_x),num_p_grip_y)    
    xyzswht_lower_grip = jnp.stack([x_ids,y_ids,z_ids,section_ids,widths,heights,thicknesses],axis=1)

    # Lower Fillet
    num_p_fillet_y = int(fillet_length/ideal_spacing)+1
    num_p_fillet_z = int(thickness/ideal_spacing)+1
    y = jnp.linspace(-fillet_upper_corner_y,-gage_length/2,num_p_fillet_y)
    w = (fillet_center_x - ((1 - (y+fillet_center_y)**2 / (fillet_major_diameter/2)**2) * (fillet_minor_diameter/2)**2)**0.5)*2
    for j in range(num_p_fillet_y):
        num_p_fillet_x = int(w[j]/ideal_spacing)+1
        section_ids = jnp.repeat(1,num_p_fillet_x*num_p_fillet_z)
        widths = jnp.repeat(num_p_fillet_x,num_p_fillet_x*num_p_fillet_z)
        heights = jnp.repeat(num_p_fillet_y,num_p_fillet_x*num_p_fillet_z)
        thicknesses = jnp.repeat(num_p_fillet_z,num_p_fillet_x*num_p_fillet_z)
        x_ids = jnp.repeat(jnp.arange(0,num_p_fillet_x),num_p_fillet_z)
        y_ids = jnp.repeat(j,num_p_fillet_x*num_p_fillet_z)
        z_ids = jnp.tile(jnp.arange(0,num_p_fillet_z),num_p_fillet_x)
        if j == 0:
            xyzswht_lower_fillet = jnp.stack([x_ids,y_ids,z_ids,section_ids,widths,heights,thicknesses],axis=1)
        else:
            xyzswht_lower_fillet = jnp.concatenate([xyzswht_lower_fillet,jnp.stack([x_ids,y_ids,z_ids,section_ids,widths,heights,thicknesses],axis=1)],axis=0)

    # Gage section
    num_p_gage_x = int(gage_width/ideal_spacing)+1
    num_p_gage_y = int(gage_length/ideal_spacing)-1
    num_p_gage_z = int(thickness/ideal_spacing)+1
    section_ids = jnp.repeat(2,num_p_gage_x*num_p_gage_y*num_p_gage_z)
    widths = jnp.repeat(num_p_gage_x,num_p_gage_x*num_p_gage_y*num_p_gage_z)
    heights = jnp.repeat(num_p_gage_y,num_p_gage_x*num_p_gage_y*num_p_gage_z)
    thicknesses = jnp.repeat(num_p_gage_z,num_p_gage_x*num_p_gage_y*num_p_gage_z)
    x_ids = jnp.repeat(jnp.repeat(jnp.arange(0,num_p_gage_x),num_p_gage_y),num_p_gage_z)
    y_ids = jnp.tile(jnp.repeat(jnp.arange(0,num_p_gage_y),num_p_gage_z),num_p_gage_x)
    z_ids = jnp.tile(jnp.tile(jnp.arange(0,num_p_gage_z),num_p_gage_x),num_p_gage_y)
    xyzswht_gage = jnp.stack([x_ids,y_ids,z_ids,section_ids,widths,heights,thicknesses],axis=1)

    # Upper Fillet
    num_p_fillet_y = int(fillet_length/ideal_spacing)+1
    num_p_fillet_z = int(thickness/ideal_spacing)+1
    y = jnp.linspace(gage_length/2,gage_length/2 + fillet_length,num_p_fillet_y)
    w = (fillet_center_x - ((1 - (y-fillet_center_y)**2 / (fillet_major_diameter/2)**2) * (fillet_minor_diameter/2)**2)**0.5)*2
    for j in range(num_p_fillet_y):
        num_p_fillet_x = int(w[j]/ideal_spacing)+1
        section_ids = jnp.repeat(3,num_p_fillet_x*num_p_fillet_z)
        widths = jnp.repeat(num_p_fillet_x,num_p_fillet_x*num_p_fillet_z)
        heights = jnp.repeat(num_p_fillet_y,num_p_fillet_x*num_p_fillet_z)
        thicknesses = jnp.repeat(num_p_fillet_z,num_p_fillet_x*num_p_fillet_z)
        x_ids = jnp.repeat(jnp.arange(0,num_p_fillet_x),num_p_fillet_z)
        y_ids = jnp.repeat(j,num_p_fillet_x*num_p_fillet_z)
        z_ids = jnp.tile(jnp.arange(0,num_p_fillet_z),num_p_fillet_x)
        if j == 0:
            xyzswht_upper_fillet = jnp.stack([x_ids,y_ids,z_ids,section_ids,widths,heights,thicknesses],axis=1)
        else:
            xyzswht_upper_fillet = jnp.concatenate([xyzswht_upper_fillet,jnp.stack([x_ids,y_ids,z_ids,section_ids,widths,heights,thicknesses],axis=1)],axis=0)

    # Upper Grip
    num_p_grip_x = int(grip_width/ideal_spacing)+1
    num_p_grip_y = int(grip_length/ideal_spacing)
    num_p_grip_z = int(thickness/ideal_spacing)+1
    section_ids = jnp.repeat(4,num_p_grip_x*num_p_grip_y*num_p_grip_z)
    widths = jnp.repeat(num_p_grip_x,num_p_grip_x*num_p_grip_y*num_p_grip_z)
    heights = jnp.repeat(num_p_grip_y,num_p_grip_x*num_p_grip_y*num_p_grip_z)
    thicknesses = jnp.repeat(num_p_grip_z,num_p_grip_x*num_p_grip_y*num_p_grip_z)
    x_ids = jnp.repeat(jnp.repeat(jnp.arange(0,num_p_grip_x),num_p_grip_y),num_p_grip_z)
    y_ids = jnp.tile(jnp.repeat(jnp.arange(0,num_p_grip_y),num_p_grip_z),num_p_grip_x)
    z_ids = jnp.tile(jnp.tile(jnp.arange(0,num_p_grip_z),num_p_grip_x),num_p_grip_y)    
    xyzswht_upper_grip = jnp.stack([x_ids,y_ids,z_ids,section_ids,widths,heights,thicknesses],axis=1)

    return jnp.concatenate([xyzswht_lower_grip,xyzswht_lower_fillet,xyzswht_gage,xyzswht_upper_fillet,xyzswht_upper_grip],axis=0) # xyzswht_upper_grip

@jax.jit
def generate_tensile_bar_positions(grip_width:            float,
                                   grip_length:           float,
                                   gage_width:            float,
                                   gage_length:           float,
                                   thickness:             float,
                                   fillet_major_diameter: float,
                                   fillet_minor_diameter: float,
                                   xyzswht:               jnp.array
                                   ) -> jnp.array:
    fillet_center_x = fillet_minor_diameter/2 + gage_width/2
    fillet_center_y = gage_length/2
    fillet_length = ((1 - (grip_width/2 - fillet_minor_diameter/2 - gage_width/2)**2 / (fillet_minor_diameter/2)**2) * (fillet_major_diameter/2)**2)**0.5
    fillet_width_upper = lambda y: 2*(fillet_center_x - ((1 - (y-fillet_center_y)**2 / (fillet_major_diameter/2)**2) * (fillet_minor_diameter/2)**2)**0.5)
    fillet_width_lower = lambda y: 2*(fillet_center_x - ((1 - (y+fillet_center_y)**2 / (fillet_major_diameter/2)**2) * (fillet_minor_diameter/2)**2)**0.5)

    def xyz_lower_grip(id):
        x_f = id[0]/id[4]
        y_f = id[1]/id[5]
        z_f = id[2]/id[6]
        x = grip_width * (x_f - 0.5)
        y = grip_length * (y_f - 1) - gage_length/2 - fillet_length
        z = thickness * (z_f - 0.5)
        return jnp.array([x,y,z])
    
    def xyz_lower_fillet(id):
        x_f = id[0]/id[4]
        y_f = id[1]/(id[5]-1)
        z_f = id[2]/id[6]
        y = fillet_length * (y_f - 1) - gage_length/2
        fillet_width = fillet_width_lower(y) 
        x = fillet_width * (x_f - 0.5)
        z = thickness * (z_f - 0.5)
        return jnp.array([x,y,z])
    
    def xyz_gage(id):
        x_f = id[0]/id[4]
        y_f = (id[1]+1)/(id[5]+1)
        z_f = id[2]/id[6] 
        x = gage_width * (x_f - 0.5)
        y = gage_length * (y_f) - gage_length/2
        z = thickness * (z_f - 0.5)
        return jnp.array([x,y,z])
    
    def xyz_upper_fillet(id):
        x_f = id[0]/id[4]
        y_f = id[1]/(id[5]-1)
        z_f = id[2]/id[6]
        y = fillet_length * (y_f) + gage_length/2
        fillet_width = fillet_width_upper(y) 
        x = fillet_width * (x_f - 0.5)
        z = thickness * (z_f - 0.5)
        return jnp.array([x,y,z])
    
    def xyz_upper_grip(id):
        x_f = id[0]/id[4]
        y_f = (id[1]+1)/id[5]
        z_f = id[2]/id[6]
        x = grip_width * (x_f - 0.5)
        y = grip_length * (y_f) + gage_length/2 + fillet_length
        z = thickness * (z_f - 0.5)
        return jnp.array([x,y,z])

    evals = [
        xyz_lower_grip,
        xyz_lower_fillet,
        xyz_gage,
        xyz_upper_fillet,
        xyz_upper_grip
    ]

    xyz_positions = jnp.vectorize(jax.lax.switch,excluded=(1,),signature='(),(n)->(m)')(xyzswht[:,3], evals, xyzswht)
    
    return xyz_positions