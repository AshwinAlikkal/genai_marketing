system_prompt_for_characteristic_prompt = f"""
    You will be provided with a json file format of a data. 
    You are required to return the output in the below format in the form of string. 

    The format is given below:
        Demographics:
            - Age: <Age group> <Has to be one particular group>
            - Income: <Income group of that particular age group>
            - Location: <Location of that particular group>
            - Family Status: <Marital Status along with no. of children for this particular group>

        Psychographics:
            - <What kind of products do these people buy from the data and why do they do that (ANSWER BASED ON YOUR UNDERSTANDING)?> 
            (YOU HAVE TO ANSWER THIS FROM THE DATA BASED ON YOUR OWN UNDERSTANDING from the data and in 1 line)

        Behavioral Characteristics:
            - <What top 2 products do they buy mostly from the data and what they dont buy?> 
            (YOU HAVE TO ANSWER THIS FROM THE DATA BASED ON YOUR OWN UNDERSTANDING from the data and in 1 line)

        Preferred marketing Channels:
          - <HOW CAN WE APPROACH THEM, i,e. though web, catalog, or store?>  
          (YOU HAVE TO ANSWER THIS FROM THE DATA BASED ON YOUR OWN UNDERSTANDING from the data and in 1 line)


    You have to exactly follow the above format and your main instructions are in capital letters. 
    Also only FILTER the age group and income based on your understanding (LIKE YOU DO THE FILTER IN PYTHON) 
    and make sure that you are answering the rest based on that.
    ALSO I DONT WANT YOU TO RETURN THE OUTPUT IN PYTHON
    """

system_prompt_for_stable_diffusion = f"""
    You will be provided with a summary of a particular cluster.
    Now for the above, I need you to generate SUMMARIZED SITUATIONAL PROMPT taking all and each section of the summarized cluster.  
    It should be one liner in such a way that an accurate, realistic and natural image can be generated . 
    
    The situaion should be very UNIQUE and REALISTIC marketing situations where the 
    main focus will be on the demographics of the person and all the products they are buying. 
    
    Like for example: 
    Young spanish couple aged 30-40s, having lower income with atleast 1 child, 
    carefully budgeting their finances to prioritize investing in gold for their family future. 
    They are buying jewellery from the store. 
    
    The above prompt consists of demographics, behavioral characteristics and situations. 

    Like in the above example, YOU HAVE TO EXACTLY FOLLOW THE 1st LINE OF THE EXAMPLE. i.e. THE DEMOGRAPHIC PART.
    
    MAKE SURE YOU ARE NOT CHANGING THE PATTERN OF DEMOGRAPHICS AND BEHAVIORAL CHARACTERISTICS. 
    
    You should not necessarily follow same marketing patterns for situations like the case in the above example. 
    You can also provide the situations according to cluster.
    
    For example: If the people within the cluster have upper income , 
    then you can write something related to candle light dinner. 
    or maybe they are going to do a unique thing according to the product. 
    But it should not always be related to shopping. It should be like an advertisement. 
    
    If they have lower income, then they are shopping on a local market or maybe they are farming or something.
    """

additional_image_instruction = f"""\n\n Do not generate cartoonish or artisitic images.
                                                            Generate realistic and photographic images. Also dont add unnecessary texts to the images"""