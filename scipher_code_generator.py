import random
import string

def generate_unique_identifier(length=4):
    """Generate a random alphanumeric identifier of a given length."""
    characters = string.ascii_uppercase + string.digits
    return ''.join(random.choices(characters, k=length))

def assign_content_code(name, language, primary_genre, sub_genre, secondary_genre, duration):
    """
    Assign a unique code to a piece of content based on the following format:
    
        X-00-00-00-00-####
    
    Where:
      X: Language Code (H for Hindi, E for English, M for Marathi, T for Telugu)
      First 00: Primary Genre Code
      Second 00: Sub-Genre Code
      Third 00: Secondary Genre Code
      Fourth 00: Duration (formatted as two digits)
      ####: A randomly generated unique identifier (letters and digits)
    """
    # Define language codes
    language_codes = {
        "Hindi": "H",
        "English": "E",
        "Marathi": "M",
        "Telugu": "T"
    }
    
    # Define genre codes
    genre_codes = {
        "Action": "01",
        "Adventure": "02",
        "Crime": "03",
        "Fantasy": "04",
        "Motivation": "05",
        "Documentary": "06",
        "Finance": "07",
        "Mystery": "08",
        "Suspense": "09",
        "Thriller": "10",
        "Drama": "11",
        "Romance": "12",
        "Coming-of-age": "13",
        "Family": "14",
        "Classic": "15",
        "Historical": "16",
        "Religion": "17",
        "Patriotic": "18",
        "Comedy": "19"
    }
    
    # Retrieve the corresponding codes. If not found, default to "XX" or "00"
    lang_code = language_codes.get(language, "XX")
    primary_code = genre_codes.get(primary_genre, "00")
    sub_code = genre_codes.get(sub_genre, "00")
    secondary_code = genre_codes.get(secondary_genre, "00")
    
    # Format the duration as a two-digit string
    duration_code = f"{int(duration):02d}"
    
    # Generate a unique identifier
    unique_identifier = generate_unique_identifier()
    
    # Construct the final content code
    content_code = f"{lang_code}-{primary_code}-{sub_code}-{secondary_code}-{duration_code}-{unique_identifier}"
    return content_code

if __name__ == "__main__":
    # Accept input from the user
    name = input("Enter the name of the content: ").strip()
    language = input("Enter the language (Hindi, English, Marathi, Telugu): ").strip()
    primary_genre = input("Enter the primary genre: ").strip()
    sub_genre = input("Enter the sub-genre: ").strip()
    secondary_genre = input("Enter the secondary genre: ").strip()
    duration = input("Enter the duration (numeric value): ").strip()
    
    code = assign_content_code(name, language, primary_genre, sub_genre, secondary_genre, duration)
    print(f"\nThe unique code for '{name}' is: {code}")