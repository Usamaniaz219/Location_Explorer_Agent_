import requests
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
import webbrowser
import joblib
from shapely.geometry import Point
from folium.plugins import MarkerCluster
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (classification_report, mean_squared_error, 
                            r2_score, roc_auc_score, mean_absolute_error)
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import os

# SMOTE implementation with fallback
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    from sklearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = False
    print("Note: imblearn package not found. SMOTE sampling will be disabled.")

warnings.filterwarnings('ignore', category=UserWarning)

class EnhancedLocationExplorerAgent:
    STATE_MAPPING = {
        'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR',
        'california': 'CA', 'colorado': 'CO', 'connecticut': 'CT',
        'delaware': 'DE', 'florida': 'FL', 'georgia': 'GA', 'hawaii': 'HI',
        'idaho': 'ID', 'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA',
        'kansas': 'KS', 'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME',
        'maryland': 'MD', 'massachusetts': 'MA', 'michigan': 'MI',
        'minnesota': 'MN', 'mississippi': 'MS', 'missouri': 'MO',
        'montana': 'MT', 'nebraska': 'NE', 'nevada': 'NV',
        'new hampshire': 'NH', 'new jersey': 'NJ', 'new mexico': 'NM',
        'new york': 'NY', 'north carolina': 'NC', 'north dakota': 'ND',
        'ohio': 'OH', 'oklahoma': 'OK', 'oregon': 'OR', 'pennsylvania': 'PA',
        'rhode island': 'RI', 'south carolina': 'SC', 'south dakota': 'SD',
        'tennessee': 'TN', 'texas': 'TX', 'utah': 'UT', 'vermont': 'VT',
        'virginia': 'VA', 'washington': 'WA', 'west virginia': 'WV',
        'wisconsin': 'WI', 'wyoming': 'WY',
        'district of columbia': 'DC'
    }
    
    ZONING_API_KEY = "852fba3cae77d1533c563d58d6ad03e32d5d1e70"
    ZONING_API_BASE = "https://public-api-test.zoneomics.com/v2"

    def __init__(self, data_path, model_path=None, regression_model_path=None):
        self.utilization_threshold = 0.1
        self.model = None
        self.regression_model = None
        self.feature_columns = None

        self.df = pd.read_excel(data_path)
        print("df shape before",self.df.shape)
        self._preprocess_data()
        print("df shape after preprocessing",self.df.shape)

        if model_path and regression_model_path:
            self.load_model(model_path)
            self.load_regression_model(regression_model_path)
        else:
            self.train_regression_model()

        print("🚀 Enhanced Location Explorer Agent initialized with suitable use filtering!")

    def _preprocess_data(self):
        """Enhanced data cleaning with robust NaN handling"""
        # Convert and clean numerical fields
        num_cols = ['Construction Date', 'Available Square Feet', 'Building Rentable Square Feet']
        self.df[num_cols] = self.df[num_cols].apply(pd.to_numeric, errors='coerce')
        
        # Handle zeros and percentages with robust NaN handling
        with np.errstate(divide='ignore', invalid='ignore'):
            self.df['Available Space %'] = ((self.df['Available Square Feet'] / 
                                          self.df['Building Rentable Square Feet'].replace(0, np.nan)) * 100)
        self.df['Available Space %'] = self.df['Available Space %'].clip(0, 100)
        
        # Enhanced categorical cleaning with NaN handling
        self.df['Owned or Leased'] = (self.df['Owned or Leased']
                                     .fillna('Unknown')
                                     .str.strip()
                                     .str.upper()
                                     .map({'L': 'LEASED', 'O': 'OWNED', 'F': 'OWNED'})
                                     .fillna('UNKNOWN'))
        
        # State abbreviation mapping with NaN handling
        self.df['State'] = (self.df['State']
                           .str.lower()
                           .map(self.STATE_MAPPING)
                           .fillna(self.df['State']))
        
        # Target variables
        self.df['Underutilized'] = (self.df['Available Space %'] > self.utilization_threshold).astype(int)
        self.df['Has_Available_Space'] = (self.df['Available Square Feet'] > 0).astype(int)
        
        # Create building age feature with NaN handling
        current_year = datetime.now().year
        self.df['Building_Age'] = current_year - self.df['Construction Date'].fillna(current_year)
        
        # Create size categories with NaN handling
        self.df['Size_Category'] = pd.cut(
            self.df['Building Rentable Square Feet'].fillna(0),
            bins=[0, 10000, 50000, 200000, np.inf],
            labels=['small', 'medium', 'large', 'xlarge']
        ).astype(str)
        
        # Geo processing with NaN handling
        valid_coords = self.df[['Longitude', 'Latitude']].notna().all(axis=1)
        self.df.loc[valid_coords, 'geometry'] = self.df[valid_coords].apply(
            lambda row: Point(row['Longitude'], row['Latitude']), axis=1
        )
        self.gdf = gpd.GeoDataFrame(self.df, geometry='geometry', crs="EPSG:4326")

    def _prepare_features(self, df=None):
        """Feature engineering with robust categorical handling"""
        if df is None:
            df = self.df.copy()

        print("dataframe shape",df.shape) 
        features = [
            'Building Rentable Square Feet',
            'Construction Date',
            'Building_Age',
            'Owned or Leased',
            'GSA Region',
            'State',
            'Building Status',
            'Real Property Asset Type',
            'Size_Category',
            'Latitude',
            'Longitude'
        ]
        
        X = df[features].copy()
        
        # Convert all categorical columns to strings
        categorical_cols = ['Owned or Leased', 'GSA Region', 'State',
                          'Building Status', 'Real Property Asset Type',
                          'Size_Category']
        
        for col in categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype(str).fillna('Unknown')
        
        # One-hot encoding
        X = pd.get_dummies(X, columns=categorical_cols)
        
        # Align columns if feature set exists
        if self.feature_columns is not None:
            missing_cols = set(self.feature_columns) - set(X.columns)
            for col in missing_cols:
                X[col] = 0
            X = X[self.feature_columns]
        
        return X

    def _filter_suitable_uses(self, permitted_uses, current_use):
        """Filter permitted uses to retain only those suitable for vacant building spaces"""
        if not isinstance(permitted_uses, list):
            return []
        
        # Define suitable use categories for vacant building spaces
        suitable_uses = {
            'general retail', 'professional services', 'personal services',
            'eating and drinking establishments', 'professional office',
            'public, quasi-public, and government', 'schools and universities',
            'research and laboratory facilities', 'medical',
            'equipment rental and repair services (excluding automobiles)',
            'contractors office', 'storage and warehouses',
            'communication towers', 'multi family dwelling',
            'communal housing', 'hotel/motel', 'parking garage or lot'
        }
        
        # Also include uses similar to current use
        if isinstance(current_use, str):
            similar_uses = {current_use.lower()}
        else:
            similar_uses = set()
        
        # Filter and return only suitable uses
        return [use for use in permitted_uses 
                if use.lower() in suitable_uses or use.lower() in similar_uses]

    def is_office_recommended(self, property_data):
        """
        Check if converting to office is recommended based on:
        - Available square footage (min. 5,000 sqft for offices)
        - Zoning permissions
        - Building characteristics (age, current use)
        Returns dict with verdict and details
        """
        zoning_info = {
            'permitted_uses': property_data.get('permitted_uses', []),
            'zone_type': property_data.get('zone_type', '')
        }
        
        analysis = {
            'minimum_sqft_met': False,
            'zoning_allowed': False,
            'current_use_compatible': False,
            'recommended': False,
            'reasons': []
        }
        
        # 1. Check available space (minimum 5,000 sqft for offices)
        available_sqft = property_data.get('Available Square Feet', 0)
        if available_sqft >= 5000:
            analysis['minimum_sqft_met'] = True
            analysis['reasons'].append(f" Sufficient space ({available_sqft:,.0f} sqft)")
        else:
            analysis['reasons'].append(f" Insufficient space ({available_sqft:,.0f} sqft < 5,000 minimum)")
        
        # 2. Check zoning
        office_allowed = 'professional office' in [use.lower() for use in zoning_info.get('permitted_uses', [])]
        analysis['zoning_allowed'] = office_allowed
        if office_allowed:
            analysis['reasons'].append(" Zoning permits offices")
        else:
            analysis['reasons'].append(" Zoning prohibits offices")
        
        # 3. Check building compatibility (current use and age)
        current_use = str(property_data.get('Real Property Asset Type', '')).lower()
        compatible_uses = ['office', 'commercial', 'retail', 'government', 'education']
        analysis['current_use_compatible'] = any(x in current_use for x in compatible_uses)
        
        building_age = property_data.get('Building_Age', 0)
        if analysis['current_use_compatible']:
            analysis['reasons'].append(f" Compatible current use ({current_use})")
        else:
            analysis['reasons'].append(f" Current use may require renovation ({current_use})")
        
        # Final recommendation (all conditions must be met)
        analysis['recommended'] = all([
            analysis['minimum_sqft_met'],
            analysis['zoning_allowed'],
            building_age < 50  # Prefer buildings <50 years old
        ])
        
        if analysis['recommended']:
            analysis['reasons'].append(" STRONG RECOMMENDATION for office conversion")
        else:
            analysis['reasons'].append(" Not ideal for office conversion")
        
        return analysis


    def train_regression_model(self, test_size=0.1, random_state=11):
        """Combined model training for both classification and regression"""
        print("Training combined classification and regression models...")
        
        # Prepare features
        X = self._prepare_features()
        
        # Classification model for Underutilized
        y_class = self.df['Underutilized']
        valid_rows = y_class.notna()
        X_class = X[valid_rows]
        y_class = y_class[valid_rows]
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_class, y_class, test_size=test_size, random_state=random_state, stratify=y_class
        )
        
        # Create the classification model with Random Forest
        base_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1
        )
        
        if SMOTE_AVAILABLE:
            self.model = ImbPipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('smote', SMOTE(random_state=random_state)),
                ('classifier', CalibratedClassifierCV(base_classifier, cv=5, method='isotonic'))
            ])
        else:
            self.model = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('classifier', CalibratedClassifierCV(base_classifier, cv=5, method='isotonic'))
            ])
        
        self.model.fit(X_train, y_train)
        
        # Store feature columns for later use
        self.feature_columns = X.columns.tolist()
        
        # Classification evaluation
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        print("\nClassification Model Evaluation (Underutilized):")
        print(classification_report(y_test, y_pred))
        print(f"Brier Score: {np.mean((y_proba - y_test) ** 2):.4f}")
        print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
        
        # Regression model for Available Square Feet
        non_zero = self.df[(self.df['Has_Available_Space'] == 1) & 
                          (self.df['Available Square Feet'].notna())]
        
        if len(non_zero) > 0:
            X_nz = self._prepare_features(non_zero)
            y_nz = non_zero['Available Square Feet']
            
            self.regression_model = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('regressor', RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_leaf=5,
                    random_state=random_state,
                    n_jobs=-1
                ))
            ])
            
            # Cross-validation
            scores = cross_val_score(
                self.regression_model,
                X_nz,
                y_nz,
                cv=5,
                scoring='r2'
            )
            print(f"\nCross-validated R² scores for regression: {scores}")
            
            self.regression_model.fit(X_nz, y_nz)
            
            # Evaluate regression
            X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
                X_nz, y_nz, test_size=test_size, random_state=random_state
            )
            y_pred_reg = self.regression_model.predict(X_test_reg)
            
            print("\nRegression Model Evaluation (Available Square Feet):")
            print(f"MSE: {mean_squared_error(y_test_reg, y_pred_reg):.2f}")
            print(f"RMSE: {np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)):.2f}")
            print(f"R²: {r2_score(y_test_reg, y_pred_reg):.2f}")
            print(f"MAE: {mean_absolute_error(y_test_reg, y_pred_reg):.2f}")
        
        return {
            'classification_model': self.model,
            'regression_model': self.regression_model
        }

    def get_office_recommended_buildings(self, lat=None, lng=None, radius_km=None):
        """
        Returns only buildings where office conversion is recommended.
        Can filter by location if coordinates and radius are provided.
        
        Args:
            lat (float): Latitude for location-based filtering (optional)
            lng (float): Longitude for location-based filtering (optional)
            radius_km (float): Radius in kilometers for location-based filtering (optional)
        
        Returns:
            GeoDataFrame: Buildings with office conversion recommendation
        """
        if lat is not None and lng is not None and radius_km is not None:
            # Get underutilized properties in the specified area
            underutilized_gdf = self.analyze_underutilized_spaces(lat, lng, radius_km)
        else:
            # Use all properties in the dataset
            underutilized_gdf = self.gdf.copy()
            # Prepare features and predict underutilization for all properties
            X = self._prepare_features()
            underutilized_gdf['Predicted_Underutilized'] = self.model.predict(X)
            underutilized_gdf['Underutilized_Probability'] = self.model.predict_proba(X)[:, 1]
            # Filter for predicted underutilized properties
            underutilized_gdf = underutilized_gdf[
                (underutilized_gdf['Predicted_Underutilized'] == 1) & 
                (underutilized_gdf['Underutilized_Probability'] > 0.7)
            ].copy()
        
        if underutilized_gdf is None or underutilized_gdf.empty:
            print("No underutilized properties found")
            return None
        
        office_recommended = []
        
        for _, row in underutilized_gdf.iterrows():
            office_analysis = self.is_office_recommended(row)
            if office_analysis['recommended']:
                # Add office recommendation details to the row
                recommended_row = row.copy()
                recommended_row['office_recommendation_reasons'] = " | ".join(office_analysis['reasons'])
                office_recommended.append(recommended_row)
        
        if not office_recommended:
            print("No buildings with office conversion recommendation found")
            return None
        
        # Create a GeoDataFrame from the recommended buildings
        office_gdf = gpd.GeoDataFrame(office_recommended, geometry='geometry', crs=underutilized_gdf.crs)
        
        # Select relevant columns for output
        output_columns = [
            'Real Property Asset Name', 'Street Address', 'City', 'State',
            'Available Square Feet', 'Available Space %', 'Building Rentable Square Feet',
            'Building_Age', 'Real Property Asset Type', 'zone_type', 'permitted_uses',
            'office_recommendation_reasons', 'geometry'
        ]
        
        # Only include columns that exist in the DataFrame
        available_columns = [col for col in output_columns if col in office_gdf.columns]
        office_gdf = office_gdf[available_columns]
        
        return office_gdf

    def get_zoning_info(self, lat, lng):
        """Fetch zoning information for a specific location"""
        url = f"{self.ZONING_API_BASE}/zoneDetail"
        params = {
            'api_key': self.ZONING_API_KEY,
            'lat': lat,
            'lng': lng
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    return data.get('data')
            print(f"Zoning API response: {response.status_code} - {response.text}")
            return None
        except Exception as e:
            print(f"Error fetching zoning info: {e}")
            return None

    def get_permitted_uses(self, lat, lng, address=None):
        """Fetch permitted land uses for a specific location"""
        url = f"{self.ZONING_API_BASE}/landuses"
        params = {
            'api_key': self.ZONING_API_KEY,
            'lat': lat,
            'lng': lng
        }
        if address:
            params['address'] = address
            
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    return data.get('data')
            print(f"Landuses API response: {response.status_code} - {response.text}")
            return None
        except Exception as e:
            print(f"Error fetching permitted uses: {e}")
            return None

   

    def analyze_underutilized_spaces(self, center_lat, center_lng, radius_km=1.5):
        """Find underutilized spaces within radius and analyze zoning information"""
        if not hasattr(self, 'gdf'):
            self.gdf = gpd.GeoDataFrame(
                self.df,
                geometry=gpd.points_from_xy(self.df.Longitude, self.df.Latitude),
                crs="EPSG:4326"
            )
        
        # Convert to projected CRS for accurate distance calculation
        gdf_projected = self.gdf.to_crs("EPSG:3857")
        center_point = gpd.GeoSeries([Point(center_lng, center_lat)], crs="EPSG:4326").to_crs("EPSG:3857")[0]
        
        # Calculate distances
        gdf_projected['distance_m'] = gdf_projected.geometry.distance(center_point)
        gdf_projected['distance_km'] = gdf_projected['distance_m'] / 1000

        print("shape after gdf_ ...",gdf_projected.shape)
        
        # Filter by distance
        nearby_properties = gdf_projected[gdf_projected['distance_km'] <= radius_km].copy()
        print(nearby_properties.shape)
        if nearby_properties.empty:
            print(f"No properties found within {radius_km}km of the specified location")
            return None
        
        # Prepare features with proper categorical handling
        X_nearby = self._prepare_features(nearby_properties)
        print("X nearby shape is ",X_nearby.shape)
        # Get predictions
        nearby_properties['Predicted_Underutilized'] = self.model.predict(X_nearby)
        nearby_properties['Underutilized_Probability'] = self.model.predict_proba(X_nearby)[:, 1]
        print("nearby properties shape ",nearby_properties.shape)
        # Filter underutilized properties
        underutilized = nearby_properties[
            (nearby_properties['Predicted_Underutilized'] == 1) & 
            (nearby_properties['Underutilized_Probability'] > 0.7)
        ].copy()
        
        if underutilized.empty:
            print("No underutilized properties found in the area")
            return None
        
        # Fetch zoning info for each underutilized property
        zoning_data = []
        for _, row in underutilized.iterrows():
            zoning_info = self.get_zoning_info(row['Latitude'], row['Longitude'])
            permitted_uses = self.get_permitted_uses(row['Latitude'], row['Longitude'], 
                                                   f"{row['Street Address']}, {row['City']}, {row['State']}")
            # plus_info = self.get_zoneomics_plus_info(row['Latitude'], row['Longitude'])
            
            zoning_data.append({
                'zone_code': zoning_info.get('zone_details', {}).get('zone_code') if zoning_info else None,
                'zone_name': zoning_info.get('zone_details', {}).get('zone_name') if zoning_info else None,
                'zone_type': zoning_info.get('zone_details', {}).get('zone_type') if zoning_info else None,
                'permitted_uses': permitted_uses if permitted_uses else [],
               
                'city_name': zoning_info.get('meta', {}).get('city_name') if zoning_info else None
            })
        
        # Add zoning info to the DataFrame
        zoning_df = pd.DataFrame(zoning_data)
        print("zoning dataframe ...",zoning_df)
        # Convert categorical columns to strings before concatenation
        for col in underutilized.select_dtypes(include=['category']).columns:
            underutilized[col] = underutilized[col].astype(str)
        
        result = pd.concat([underutilized.reset_index(drop=True), zoning_df], axis=1)
        
        # Convert back to GeoDataFrame
        result_gdf = gpd.GeoDataFrame(result, geometry='geometry', crs="EPSG:3857")
        print("result geo dataframe shape",result_gdf)
        
        return result_gdf.to_crs("EPSG:4326")

    def generate_recommendations(self, underutilized_gdf):
        """Generate intelligent recommendations with filtered suitable uses"""
        if underutilized_gdf is None or underutilized_gdf.empty:
            return None
        
        recommendations = []
        
        for _, row in underutilized_gdf.iterrows():
            # Basic info
            rec = {
                'property_name': row['Real Property Asset Name'],
                'address': f"{row['Street Address']}, {row['City']}, {row['State']}",
                'available_space': row['Available Square Feet'],
                'utilization_pct': row['Available Space %'],
                'probability_underutilized': row.get('Underutilized_Probability', 0),
                'zone_type': row.get('zone_type', 'Unknown'),
                'current_use': row['Real Property Asset Type'],
                'development_potential': row.get('plus_development_potential', 'N/A'),
                'rental_value': row.get('plus_rental_value', 'N/A')
            }
            
            # Get and filter permitted uses
            permitted_uses = row.get('permitted_uses', [])
            rec['potential_uses'] = self._filter_suitable_uses(
                permitted_uses, 
                row['Real Property Asset Type']
            )
            
            # Run specialized recommendation checks
            office_analysis = self.is_office_recommended(row)
            
            # Add specialized recommendations
            rec['office_recommendation'] = office_analysis['recommended']
            rec['office_reasons'] = " | ".join(office_analysis['reasons'])
            
            
            # Enhanced recommendation logic with filtered uses
            recommendations_text = []
            
            # Base recommendations on utilization
            if row['Available Space %'] > 70:
                recommendations_text.append("Highly underutilized (70%+ available) - strong candidate for redevelopment")
            elif row['Available Space %'] > 40:
                recommendations_text.append("Significantly underutilized (40%+ available) - consider repurposing")
            
            # Add specialized recommendations if strong
            if office_analysis['recommended']:
                recommendations_text.append(" Strong office conversion potential")
            
            # Recommendations based on PLUS data
            dev_potential = row.get('plus_development_potential')
            if dev_potential and isinstance(dev_potential, str):
                if dev_potential.lower() == 'high':
                    recommendations_text.append("High development potential - ideal for redevelopment")
                elif dev_potential.lower() == 'medium':
                    recommendations_text.append("Moderate development potential - good for renovation")
            
            rental_value = row.get('plus_rental_value')
            if rental_value and isinstance(rental_value, str):
                if rental_value.lower() == 'high':
                    recommendations_text.append("High rental value potential - consider income-generating uses")
                elif rental_value.lower() == 'medium':
                    recommendations_text.append("Moderate rental value - consider stable tenant uses")
            
            # Age-based recommendations
            if row['Building_Age'] > 50 and row['Available Space %'] > 30:
                recommendations_text.append("Older building with significant available space - consider renovation")
            
            # Fallback recommendation if none others apply
            if not recommendations_text:
                # if filtered_uses:
                #     recommendations_text.append(f"Consider converting to: {filtered_uses[0]}")
                # :
                recommendations_text.append("Evaluate for better utilization based on zoning")
            
            rec['recommendation'] = " | ".join(recommendations_text)
            recommendations.append(rec)
        
        return pd.DataFrame(recommendations)

    def visualize_underutilized_spaces(self, underutilized_gdf, output_file="output/underutilized_spaces.html"):
        """Create an interactive map showing underutilized spaces with office recommendations highlighted"""
        if underutilized_gdf is None or underutilized_gdf.empty:
            print("No underutilized spaces to visualize")
            return None
        
        # Create base map centered on the first property
        center_lat = underutilized_gdf.iloc[0]['Latitude']
        center_lng = underutilized_gdf.iloc[0]['Longitude']
        m = folium.Map(location=[center_lat, center_lng], zoom_start=14)
        
        # Create separate feature groups for different recommendation types
        fg_office_recommended = folium.FeatureGroup(name=' Office Recommended')
        fg_other = folium.FeatureGroup(name='Other Underutilized')
        marker_cluster_office = MarkerCluster().add_to(fg_office_recommended)
        marker_cluster_other = MarkerCluster().add_to(fg_other)
        
        # Add each property to the appropriate feature group
        for _, row in underutilized_gdf.iterrows():
            # Determine if office is recommended
            office_rec = self.is_office_recommended(row)
            
            # Filter permitted uses to only show suitable ones
            permitted_uses = row.get('permitted_uses', [])
            filtered_uses = self._filter_suitable_uses(permitted_uses, row['Real Property Asset Type'])
            
            # Create popup content
            popup_content = f"""
            <b>{row['Real Property Asset Name']}</b><br>
            <b>Address:</b> {row['Street Address']}, {row['City']}, {row['State']}<br>
            <b>Available Space:</b> {row['Available Square Feet']:,.0f} sqft ({row['Available Space %']:.1f}%)<br>
            <b>Current Use:</b> {row['Real Property Asset Type']}<br>
            <hr>
            <b>Zoning:</b> {row.get('zone_name', 'N/A')} ({row.get('zone_code', 'N/A')})<br>
            <b>Zone Type:</b> {row.get('zone_type', 'Unknown')}<br>
            <b>Suitable Potential Uses:</b> {', '.join(filtered_uses) if filtered_uses else 'N/A'}<br>
            <hr>
            <b>Office Conversion:</b> {" <strong>RECOMMENDED</strong>" if office_rec['recommended'] else "⚠️ Not Recommended"}<br>
            <small>{' | '.join(office_rec['reasons'])}</small>
            """
            
            # Customize marker based on office recommendation
            if office_rec['recommended']:
                # Special marker for office-recommended buildings
                icon = folium.Icon(
                    color='green',
                    icon='building',
                    prefix='fa',
                    icon_color='white'
                )
                marker = folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    popup=folium.Popup(popup_content, max_width=300),
                    icon=icon,
                    tooltip=f" {row['Real Property Asset Name']} - Office Recommended",
                )
                marker.add_to(marker_cluster_office)
            else:
                # Standard marker for other underutilized buildings
                zone_type = row.get('zone_type', 'Unknown')
                if zone_type == 'Commercial':
                    color = 'blue'
                elif zone_type == 'Residential':
                    color = 'purple'
                elif zone_type == 'Industrial':
                    color = 'orange'
                else:
                    color = 'gray'
                
                marker = folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    popup=folium.Popup(popup_content, max_width=300),
                    icon=folium.Icon(color=color, icon='building', prefix='fa'),
                    tooltip=f"{row['Real Property Asset Name']} - {row['Available Space %']:.1f}% available",
                )
                marker.add_to(marker_cluster_other)
        
        # Add feature groups to map with different colors
        fg_office_recommended.add_to(m)
        fg_other.add_to(m)
        
        # Add layer control to toggle feature groups
        folium.LayerControl(collapsed=False).add_to(m)
        
        # Add title to the map
        title_html = '''
            <h3 align="center" style="font-size:16px">
            <b>Underutilized Properties with Office Conversion Recommendations</b>
            </h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Add legend
        legend_html = '''
            <div style="position: fixed; 
                        bottom: 50px; left: 50px; width: 150px; height: 120px; 
                        border:2px solid grey; z-index:9999; font-size:12px;
                        background-color:white;
                        opacity: 0.85;
                        ">
                <div style="padding: 5px;">
                    <p style="margin: 0 0 5px;"><i class="fa fa-building" style="color: green"></i> Office Recommended</p>
                    <p style="margin: 0 0 5px;"><i class="fa fa-building" style="color: blue"></i> Commercial Zone</p>
                    <p style="margin: 0 0 5px;"><i class="fa fa-building" style="color: purple"></i> Residential Zone</p>
                    <p style="margin: 0 0 5px;"><i class="fa fa-building" style="color: orange"></i> Industrial Zone</p>
                    <p style="margin: 0 0 5px;"><i class="fa fa-building" style="color: gray"></i> Other Zone</p>
                </div>
            </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save and open the map

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        m.save(output_file)
        webbrowser.open(output_file)
        return m

    def full_analysis_pipeline(self, lat, lng, radius_km=1.5):
        """
        Complete analysis pipeline:
        1. Find underutilized spaces
        2. Get zoning and PLUS info
        3. Generate recommendations with filtered uses
        4. Visualize results
        """
        print(f" Starting analysis for location ({lat}, {lng}) within {radius_km}km radius")
        print("🔍 Gathering Zoneomics data for the area...")
        
        # First get area-wide Zoneomics data
        area_zoning = self.get_zoning_info(lat, lng)
        # area_plus = self.get_zoneomics_plus_info(lat, lng)
        
        if area_zoning:
            print(f" Area Zone: {area_zoning.get('zone_details', {}).get('zone_name', 'Unknown')}")
        # if area_plus:
        #     print(f" Area Development Potential: {area_plus.get('development_potential', 'Unknown')}")
        
        # Step 1: Find underutilized spaces
        print("🔍 Identifying underutilized properties with model predictions...")
        underutilized_gdf = self.analyze_underutilized_spaces(lat, lng, radius_km)
        
        if underutilized_gdf is None or underutilized_gdf.empty:
            print(" No underutilized properties found")
            return None
        
        # Step 2: Generate enhanced recommendations with filtered uses
        print(" Generating intelligent recommendations with filtered suitable uses...")
        recommendations = self.generate_recommendations(underutilized_gdf)
        
        # Step 3: Visualize with filtered uses
        print(" Creating interactive map with suitable uses...")
        self.visualize_underutilized_spaces(underutilized_gdf)
        
        # Save results
        recommendations.to_csv("property_recommendations.csv", index=False)
        
        # Ensure proper data types before saving GeoJSON
        for col in underutilized_gdf.select_dtypes(include=['category']).columns:
            underutilized_gdf[col] = underutilized_gdf[col].astype(str)
        underutilized_gdf.to_file("underutilized_properties.geojson", driver='GeoJSON')
        
        print(" Analysis complete! Results saved to:")
        print("- property_recommendations.csv")
        print("- underutilized_properties.geojson")
        print("- underutilized_spaces.html (interactive map)")
        
        return {
            'area_zoning': area_zoning,
            # 'area_plus': area_plus,
            'underutilized_properties': underutilized_gdf,
            'recommendations': recommendations
        }

    def predict_utilization(self, property_data):
        """Predict utilization status for new property data"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_regression_model() first.")
            
        X = self._prepare_features(property_data)
        proba = self.model.predict_proba(X)[:, 1]
        prediction = self.model.predict(X)
        
        return {
            'prediction': prediction[0],
            'probability': proba[0],
            'underutilized': bool(prediction[0])
        }

    def predict_available_space(self, property_data):
        """Two-stage prediction for available space"""
        if self.regression_model is None:
            raise ValueError("Model not trained. Call train_regression_model() first.")
            
        X = self._prepare_features(property_data)
        
        # Stage 1: Probability of having available space (from classification model)
        has_space_prob = self.model.predict_proba(X)[:, 1][0]
        
        # Stage 2: Predict amount if likely to have space
        if has_space_prob > 0.5:
            predicted_amount = self.regression_model.predict(X)[0]
        else:
            predicted_amount = 0
            
        return {
            'probability_has_space': has_space_prob,
            'predicted_amount': predicted_amount
        }

   
    def save_model(self, path):
        """Save the trained classification model"""
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns
        }, path)

    def load_model(self, path):
        """Load a trained classification model"""
        data = joblib.load(path)
        self.model = data['model']
        self.feature_columns = data['feature_columns']

    def save_regression_model(self, path):
        """Save the trained regression model"""
        joblib.dump({
            'regression_model': self.regression_model,
            'feature_columns': self.feature_columns
        }, path)

    def load_regression_model(self, path):
        """Load trained regression model"""
        data = joblib.load(path)
        self.regression_model = data['regression_model']
        self.feature_columns = data['feature_columns']







