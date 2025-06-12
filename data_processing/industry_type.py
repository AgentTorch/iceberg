from enum import Enum

class ModifiedBLSSuperSector(str, Enum):
    """
        Adapted from https://www.bls.gov/oes/tables.htm and https://www.mynextmove.org/
    """
    AGRICULTURE_FORESTRY_FISHING_HUNTING   = "AGRICULTURE_FORESTRY_FISHING_HUNTING",
    MINING_OIL_GAS_EXTRACTION              = "MINING_OIL_GAS_EXTRACTION",
    ENERGY                                 = "ENERGY", # Custom: Energy is a sector we want to track
    CONSTRUCTION                           = "CONSTRUCTION",
    MANUFACTURING                          = "MANUFACTURING",
    RETAIL_TRADE                           = "RETAIL_TRADE", # Custom: Merged Wholesale into Retail Trade
    PACKAGING_AND_GOODS_TRANSPORTATION     = "PACKAGING_AND_GOODS_TRANSPORTATION" # Custom: Split from Transportation - but made title more verbose
    PASSENGER_TRANSPORTATION               = "PASSENGER_TRANSPORTATION" # Custom: Split from Transportation - but made title more verbose
    TECHNOLOGY_AND_SOFTWARE                = "TECHNOLOGY_AND_SOFTWARE_DEVELOPMENT" # Custom: Adapted from information - but made title more verbose
    FINANCE_AND_INSURANCE                  = "FINANCE_AND_INSURANCE"
    REAL_ESTATE_RENTAL_LEASING             = "REAL_ESTATE_RENTAL_LEASING"
    
    # Following are from Professional, scientific, and technical services
    LEGAL_SERVICES                         = "LEGAL_SERVICES" # Custom: Broken from Professional, scientific, and technical services
    ARCHITECTURAL_ENGINEERING_SERVICES     = "ARCHITECTURAL_ENGINEERING_SERVICES" # Custom: Broken from Professional, scientific, and technical services
    SCIENTIFIC_RESEARCH_AND_DEVELOPMENT    = "SCIENTIFIC_RESEARCH_AND_DEVELOPMENT" # Custom: Broken from Professional, scientific, and technical services
    MANAGEMENT_CONSULTING_SERVICES         = "MANAGEMENT_CONSULTING_SERVICES" # Custom: Broken from Professional, scientific, and technical services
    MEDIA_AND_COMMUNICATIONS_SERVICES      = "MEDIA_AND_COMMUNICATIONS_SERVICES" # Custom: Broken from Media and communications

    # Following are from Management of companies and enterprises
    MANAGEMENT                             = "MANAGEMENT"
    ADMINISTRATIVE_SUPPORT_WASTE_SERVICES  = "ADMINISTRATIVE_SUPPORT_WASTE_SERVICES" # Custom: Broken from Administrative and support services
    EDUCATIONAL_SERVICES                   = "EDUCATIONAL_SERVICES"
    HEALTHCARE                             = "HEALTHCARE"
    TRAVEL_AND_ENTERTAINMENT_SERVICES      = "TRAVEL_AND_ENTERTAINMENT_SERVICES" # Custom: Merged (Arts, entertainment, and recreation) and (Accommodation and food services) 
    GOVERNMENT                             = "GOVERNMENT" # Custom: Merged Public Administration into Government Services
    SPECIAL_INDUSTRIES                     = "SPECIAL_INDUSTRIES" # Custom: Broken from Specialized industries

    # OTHER_SERVICES_EXCEPT_PUBLIC_ADMIN     = "OTHER_SERVICES_EXCEPT_PUBLIC_ADMINISTRATION" # Custom: We dropped this from the analysis