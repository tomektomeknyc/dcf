import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import streamlit as st
from io import BytesIO
from pathlib import Path
import logging
from urllib.parse import urljoin

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
# --- DEBUG: global Damodaran download counter ---
if "damo_download_calls" not in st.session_state:
    st.session_state["damo_download_calls"] = []

# ─── Build a global session with retries + browser UA ────────────────────────
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/16.5 Safari/605.1.15"
    )
})
retries = Retry(
    total=5,
    backoff_factor=0.3,
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["GET", "POST"]
)
SESSION.mount("https://", HTTPAdapter(max_retries=retries))
SESSION.mount("http://",  HTTPAdapter(max_retries=retries))

# don’t inherit any HTTP_PROXY / HTTPS_PROXY from your shell
SESSION.trust_env = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _get_beta_urls() -> dict[str, str]:
    """
    Scrape Damodaran's datacurrent.html and return a mapping:
      { "US": url, "Europe": url, "AU_NZ": url, "Japan": url, "Emerging": url, "Global": url }
    """
    INDEX = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datacurrent.html"
    
    try:
        resp = SESSION.get(INDEX, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Find all table rows and look for beta data
        all_rows = soup.find_all("tr")
        logger.info(f"Found {len(all_rows)} table rows to search")
        
        target_row = None
        
        # Look for the row containing beta links
        for idx, row in enumerate(all_rows):
            row_text = row.get_text(strip=True).lower()
            if ("beta" in row_text and "industry" in row_text and 
                ("sector" in row_text or "total" in row_text)):
                logger.info(f"Found beta row at index {idx}: {row.get_text()[:100]}")
                target_row = row
                break
        
        if not target_row:
            raise RuntimeError("Cannot locate Damodaran betas row")

        # Extract all links from the target row
        cells = target_row.find_all("td")
        logger.info(f"Found {len(cells)} cells in target row")
        
        # Collect all links from all cells in the row
        all_links = []
        for cell in cells:
            links = cell.find_all("a", href=True)
            all_links.extend(links)
        
        logger.info(f"Found {len(all_links)} total links")
        
        # Process each link
        url_mapping = {}
        for link in all_links:
            text = link.get_text().strip().lower()
            href = link.get("href")
            full_url = urljoin(INDEX, href)
            
            logger.info(f"Processing link: '{text}' -> {full_url}")
            
            # Map text to regions - be more specific with US mapping
            if text == "us" or (text.startswith("us") and "aus" not in text):
                url_mapping["US"] = full_url
            elif "europe" in text:
                url_mapping["Europe"] = full_url
            elif "japan" in text:
                url_mapping["Japan"] = full_url
            elif ("aus" in text or "nz" in text or "canada" in text):
                url_mapping["AU_NZ"] = full_url
            elif "emerging" in text:
                url_mapping["Emerging"] = full_url
            elif "china" in text and "just china" in text:
                url_mapping["China"] = full_url
            elif "india" in text and "just india" in text:
                url_mapping["India"] = full_url
            elif "global" in text:
                url_mapping["Global"] = full_url

        available_regions = list(url_mapping.keys())
        logger.info(f"Mapped regions: {available_regions}")
        
        return url_mapping

    except Exception as e:
        logger.error(f"Error in _get_beta_urls: {e}")
        raise RuntimeError(f"Failed to scrape Damodaran beta URLs: {e}")


def _download_region_beta(region: str, dest_folder: str = "Damodaran") -> str:
    """
    Download the Damodaran beta Excel for a given region.
    Returns the local filepath.
    """
     # === DEBUG: count every attempted regional download ===
    import streamlit as st
    st.session_state.setdefault("_damo_dl_calls", 0)
    st.session_state["_damo_dl_calls"] += 1
    # st.sidebar.write(
    #     f"DEBUG: _download_region_beta('{region}') "
    #     f"call #{st.session_state['_damo_dl_calls']}"
    # )
    # === END DEBUG ===
    # Get all URLs
    urls = _get_beta_urls()
    try:
        url = urls[region]
    except KeyError:
        raise ValueError(f"No Damodaran beta URL for region '{region}'. Available: {list(urls.keys())}")

    # Ensure destination folder exists
    Path(dest_folder).mkdir(parents=True, exist_ok=True)

    # Determine filename and full path
    filename = url.split("/")[-1]
    local_path = Path(dest_folder) / filename

    # Download if missing
    if not local_path.exists():
        logger.info(f"Downloading {url} to {local_path}")
        resp = SESSION.get(url, timeout=30)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(resp.content)
        logger.info(f"Successfully downloaded {local_path}")

    return str(local_path)


def _load_beta_dataframe(region: str) -> pd.DataFrame:
    """
    Load beta data from Excel file or web source into DataFrame
    """
    try:
        # Get URLs for the region
        urls = _get_beta_urls()
        if region not in urls:
            logger.error(f"Region '{region}' not available. Available regions: {list(urls.keys())}")
            return pd.DataFrame()
        
        url = urls[region]
        logger.info(f"Loading beta data for {region} from {url}")
        
        # Try to download as Excel first
        resp = SESSION.get(url, timeout=30)
        resp.raise_for_status()
        
        # Try reading as Excel
        try:
            bio = BytesIO(resp.content)
            # Try different reading strategies for better column detection
            try:
                # First try: standard read
                df = pd.read_excel(bio, sheet_name=0)
            except:
                # Second try: skip potential header rows
                bio.seek(0)
                df = pd.read_excel(bio, sheet_name=0, header=1)
            
            # Clean up DataFrame - remove completely empty rows/columns
            df = df.dropna(how='all').dropna(how='all', axis=1)
            
            logger.info(f"Successfully loaded Excel file: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            
        except Exception as excel_error:
            logger.warning(f"Failed to parse as Excel: {excel_error}")
            # Try parsing as HTML table
            soup = BeautifulSoup(resp.content, 'html.parser')
            tables = soup.find_all('table')
            
            if not tables:
                raise Exception("No tables found in the response")
            
            # Convert first table to DataFrame
            table = tables[0]
            rows = table.find_all('tr')
            
            data = []
            headers = None
            for row in rows:
                cells = row.find_all(['td', 'th'])
                row_data = [cell.get_text().strip() for cell in cells]
                if headers is None and row_data:
                    headers = row_data
                elif row_data:
                    data.append(row_data)
            
            if headers and data:
                df = pd.DataFrame(data, columns=headers)
                logger.info(f"Successfully parsed HTML table: {df.shape}")
            else:
                raise Exception("Could not parse table data")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading beta data for {region}: {e}")
        return pd.DataFrame()


# @st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_damodaran_industry_betas(region: str) -> pd.DataFrame:
    """
    Fetch median levered betas by industry from Damodaran.
    Tries Excel read (skiprows=19, cols A+C).  If that fails,
    parses the first HTML table and pulls the same columns.
    """

    # 1) Get URL
    urls = _get_beta_urls()
    if region not in urls:
        logger.error(f"No Damodaran beta URL for region '{region}'. Available: {list(urls.keys())}")
        return pd.DataFrame()
    url = urls[region]

    # 2) Download bytes
    response = SESSION.get(url, timeout=30)
    response.raise_for_status()
    bio = BytesIO(response.content)

    # 3) Try reading from Industry Averages sheet
    df = None
    try:
        bio.seek(0)
        df = pd.read_excel(
            bio,
            sheet_name='Industry Averages',  # Use the correct sheet name
            header=7,  # Header is at row 7 based on our analysis
            usecols=[0, 3],  # Column A (Industry Name) and Column D (Average Levered Beta)
            engine="xlrd"
        )
        logger.info(f"Successfully read Industry Averages sheet")
    except Exception as e:
        logger.debug(f"Failed to read Industry Averages sheet: {e}")
        # Fallback: try reading by sheet index
        try:
            bio.seek(0)
            df = pd.read_excel(
                bio,
                sheet_name=1,  # Try sheet index 1
                header=7,
                usecols=[0, 3],
                engine="xlrd"
            )
            logger.info(f"Successfully read sheet 1 as fallback")
        except Exception as e2:
            logger.debug(f"Fallback reading also failed: {e2}")
            df = None
    
    # If Excel reading completely failed, try reading without any header constraints
    if df is None or df.empty:
        try:
            bio.seek(0)
            df_raw = pd.read_excel(bio, sheet_name=0, header=None, engine="xlrd")
            logger.info(f"Read raw Excel data: shape {df_raw.shape}")
            
            # Look for the actual data by searching for "Industry" or similar
            data_start_row = None
            for i in range(min(25, len(df_raw))):  # Check first 25 rows
                row_text = ' '.join(df_raw.iloc[i].astype(str).str.lower())
                if any(keyword in row_text for keyword in ['industry', 'sector', 'business']):
                    data_start_row = i
                    logger.info(f"Found potential header row at {i}")
                    break
            
            if data_start_row is not None:
                # Extract data starting from the found row
                df = df_raw.iloc[data_start_row:].reset_index(drop=True)
                df.columns = df.iloc[0]  # Use first row as column names
                df = df.iloc[1:].reset_index(drop=True)  # Remove header row from data
                
                # Select columns 0 and 3 (Column A and Column D as per Excel structure)
                if df.shape[1] >= 4:
                    df = df.iloc[:, [0, 3]]  # Column A (Industry) and Column D (Beta)
                elif df.shape[1] >= 2:
                    df = df.iloc[:, [0, 1]]  # Fallback to first two columns
                else:
                    logger.error("Not enough columns in Excel data")
                    return pd.DataFrame()
            else:
                logger.warning("Could not find header row in Excel data")
                return pd.DataFrame()
                
        except Exception as e:
            logger.warning(f"Excel reading completely failed for {region}: {e}")
            return pd.DataFrame()

    # 4) Rename + clean - handle variable number of columns
    if len(df.columns) >= 2:
        # Only keep first two columns and rename them
        df = df.iloc[:, [0, 1]] 
        df.columns = ["Industry", "Levered_Beta"]
    else:
        logger.error(f"Insufficient columns in data: {len(df.columns)}")
        return pd.DataFrame()
    df = (
        df
        .dropna(subset=["Industry", "Levered_Beta"])
        .assign(
            # Extract numeric portion of the Beta cell
            Levered_Beta=lambda d: pd.to_numeric(
                d["Levered_Beta"]
                  .astype(str)
                  .str.extract(r"(\d+\.?\d*)")[0],
                errors="coerce"
            ),
            # Remove any leading "20. " or "1. " serial numbers
            Industry=lambda d: (
                d["Industry"]
                  .astype(str)
                  .str.replace(r"^\d+\.\s*", "", regex=True)
                  .str.strip()
            )
        )
        # Only keep reasonable betas
        .query("0.01 <= Levered_Beta <= 5.0")
        .drop_duplicates(subset=["Industry"])
        .sort_values("Industry")
        .reset_index(drop=True)
    )

    logger.info(f"Successfully extracted {len(df)} industries for {region}")
    return df




def find_industry_beta(industry_betas_df: pd.DataFrame, company_sector: str, company_industry: str = None):
    """
    Find the most relevant industry beta for a company.
    
    Args:
        industry_betas_df: DataFrame with industry beta data
        company_sector: Company's sector
        company_industry: Company's specific industry (optional)
    
    Returns:
        Dict with matched industry name and beta value, or None if not found
    """
    if industry_betas_df.empty:
        return None
    
    # Create search terms from sector and industry
    search_terms = []
    if company_industry:
        search_terms.append(company_industry.lower())
    if company_sector:
        search_terms.append(company_sector.lower())
    
    # Try exact matches first
    for term in search_terms:
        exact_match = industry_betas_df[
            industry_betas_df['Industry'].str.lower() == term
        ]
        if not exact_match.empty:
            return {
                'industry': exact_match.iloc[0]['Industry'],
                'beta': exact_match.iloc[0]['Levered_Beta']
            }
    
    # Try partial matches
    for term in search_terms:
        partial_matches = industry_betas_df[
            industry_betas_df['Industry'].str.lower().str.contains(term, na=False)
        ]
        if not partial_matches.empty:
            return {
                'industry': partial_matches.iloc[0]['Industry'],
                'beta': partial_matches.iloc[0]['Levered_Beta']
            }
    
    # Try broader keyword matching
    keywords = []
    for term in search_terms:
        keywords.extend(term.split())
    
    for keyword in keywords:
        if len(keyword) > 3:  # Skip very short words
            keyword_matches = industry_betas_df[
                industry_betas_df['Industry'].str.lower().str.contains(keyword, na=False)
            ]
            if not keyword_matches.empty:
                return {
                    'industry': keyword_matches.iloc[0]['Industry'],
                    'beta': keyword_matches.iloc[0]['Levered_Beta']
                }
    
    return None


# def calculate_wacc_with_industry_beta(market_beta: float, industry_beta: float = None, 
#                                     risk_free_rate: float = 0.03, market_premium: float = 0.06,
#                                     debt_ratio: float = 0.3, cost_of_debt: float = 0.05, 
#                                     tax_rate: float = 0.25):
def calculate_wacc_with_industry_beta(market_beta: float, industry_beta: float, 
                                    risk_free_rate: float, market_premium: float,
                                    debt_ratio: float, cost_of_debt: float, 
                                    tax_rate: float):
    """
    Calculate WACC using both company-specific beta and industry benchmark beta.
    
    Returns:
        Dict with WACC calculations for both company and industry betas
    """
    results = {}
    
    # Company WACC using company-specific beta
    cost_of_equity_company = risk_free_rate + (market_beta * market_premium)
    wacc_company = ((1 - debt_ratio) * cost_of_equity_company) + (debt_ratio * cost_of_debt * (1 - tax_rate))
    
    results['company_wacc'] = {
        'wacc': wacc_company,
        'cost_of_equity': cost_of_equity_company,
        'beta': market_beta
    }
    
    # Industry WACC using industry benchmark beta
    if industry_beta:
        cost_of_equity_industry = risk_free_rate + (industry_beta * market_premium)
        wacc_industry = ((1 - debt_ratio) * cost_of_equity_industry) + (debt_ratio * cost_of_debt * (1 - tax_rate))
        
        results['industry_wacc'] = {
            'wacc': wacc_industry,
            'cost_of_equity': cost_of_equity_industry,
            'beta': industry_beta
        }
    
    return results



def map_folder_to_region(folder_code: str) -> str:
    """
    Map folder codes (e.g. 'US', 'DE', 'AU', 'NZ') to the keys used
    in the Damodaran beta URLs.
    """
    mapping = {
        'US': 'US',
        'DE': 'Europe',
        'NZ': 'AU_NZ',
        'AU': 'AU_NZ',
        'JP': 'Japan',
        'CN': 'China',
        'IN': 'India',
    }
    return mapping.get(folder_code.upper(), 'US')




def test_fetch_damodaran_betas():
    """Test the beta fetching functionality"""
    print("Testing Damodaran beta fetching...")
    
    # Test URL extraction
    try:
        urls = _get_beta_urls()
        print(f"Found URLs: {urls}")
        
        # Test data fetching for US region
        us_data = fetch_damodaran_industry_betas("US")
        print(f"US Beta Data Shape: {us_data.shape}")
        
        if not us_data.empty:
            print(f"Sample US industries: {us_data['Industry'].head().tolist()}")
            print(f"Sample US betas: {us_data['Levered_Beta'].head().tolist()}")
        
    except Exception as e:
        print(f"Error in test: {e}")


# Test when run directly
if __name__ == "__main__":
    test_fetch_damodaran_betas()
