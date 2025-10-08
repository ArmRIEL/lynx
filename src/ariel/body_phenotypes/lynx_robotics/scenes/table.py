def table_terrain(env_mjcf):
    '''Builds a table terrain for the simulation. Includes a square table with 4 legs and a light source.'''
    # Add the same lighting as the plane terrain
    env_mjcf.worldbody.add(
        "light",
        cutoff=100,
        diffuse=[1,1,1],
        dir=[0,0,-1.3],
        directional=True,
        exponent=1,
        pos=[0, 0, 1.3],
        specular=[0.1, 0.1, 0.1],
        castshadow=False,
    )

    # Table dimensions
    table_width = 1.0
    table_depth = 1.0
    table_height = 0.8
    table_thickness = 0.05
    leg_width = 0.1
    
    # Table top (square)
    env_mjcf.worldbody.add(
        "geom", 
        friction=[0.7, 0.1, 0.1],
        conaffinity=0, # Disable collision
        contype=0,
        condim=3,
        name="table_top",
        pos=[0, 0, table_height],
        rgba=[0.6, 0.4, 0.2, 1],  # Brown wood color
        size=[table_width/2, table_depth/2, table_thickness/2], # Revert to original box size
        type="box" # Revert to box type
    )
    
    # Table legs (4 corners)
    leg_positions = [
        [table_width/2 - leg_width/2, table_depth/2 - leg_width/2],   # Front right
        [-table_width/2 + leg_width/2, table_depth/2 - leg_width/2],  # Front left
        [table_width/2 - leg_width/2, -table_depth/2 + leg_width/2],  # Back right
        [-table_width/2 + leg_width/2, -table_depth/2 + leg_width/2]  # Back left
    ]
    
    for i, (x, y) in enumerate(leg_positions):
        env_mjcf.worldbody.add(
            "geom",
            friction=[0.7, 0.1, 0.1],
            conaffinity=0, # Disable collision
            contype=0,
            condim=3,
            name=f"table_leg_{i+1}",
            pos=[x, y, table_height/2],
            rgba=[0.5, 0.3, 0.1, 1],  # Darker brown for legs
            size=[leg_width/2, leg_width/2, table_height/2], # Revert to original box size
            type="box" # Revert to box type
    )
    
    # Optional: Add a floor plane underneath
    #     env_mjcf.worldbody.add("geom", friction=[0.7, 0.1, 0.1], conaffinity=1, condim=3, name="floor", pos=[0,0,0], rgba=[0.8, 0.9, 0.8, 1], size=[40,40,40], type="plane", material="MatPlane")

    # env_mjcf.worldbody.add(
    #     "geom", 
    #     friction=[0.7, 0.1, 0.1], 
    #     conaffinity=1, 
    #     condim=3, 
    #     name="floor", 
    #     pos=[0, 0, -0.01], 
    #     rgba=[0.8, 0.9, 0.8, 1], 
    #     size=[40, 40, 40], 
    #     type="plane",
    #     material="MatPlane"
    # )

    # Original target site (commented out)
    # env_mjcf.worldbody.add(
    #     "site",
    #     type="sphere",
    #     size=[0.1],
    #     rgba=[1, 0, 0, 1],
    #     pos=[0, 0, table_height + table_thickness/2 + 0.01],  # Slightly above the table
    #     name="target"
    # )
    # Wrap the target site in a mocap body for dynamic control in MJX
    target_mocap_body = env_mjcf.worldbody.add(
        "body", name="target_mocap_body", mocap="true", pos=[0, 0, table_height + table_thickness/2 + 0.01]
    )
    target_mocap_body.add(
        "site",
        type="sphere",
        size=[0.01], # Smaller size for the site itself, as the body defines the main position
        rgba=[1, 0, 0, 1],
        name="target",
        pos=[0,0,0] # Position relative to the mocap body
    )