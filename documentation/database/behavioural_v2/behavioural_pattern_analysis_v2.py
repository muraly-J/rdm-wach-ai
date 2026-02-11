import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import urllib3
from influxdb_client import InfluxDBClient
from dotenv import load_dotenv
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from datetime import datetime

# --- SILENCE WARNINGS ---
warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)

load_dotenv()

# Config
URL = os.getenv("url")
TOKEN = os.getenv("token")
ORG = os.getenv("org")
BUCKET = os.getenv("bucket")

# Get the directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "AHU Relational Database - Relationships.csv")

# CONFIGURATION
MAX_DEVICES_PER_CLASS = 30  # Classes with more devices will be subdivided

def normalize_timeseries(series):
    """
    Normalize time series to 0-1 range to compare PATTERN, not magnitude.
    This allows devices with different power levels but similar behaviors to match.
    """
    if series.max() == series.min():
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.min()) / (series.max() - series.min())

def calculate_pattern_similarity(ts1, ts2):
    """
    Calculate similarity between two normalized time series patterns.
    Returns a score between 0 (completely different) and 1 (identical pattern).
    Uses Pearson correlation of normalized series.
    """
    if len(ts1) != len(ts2):
        # Resample to same length if needed
        min_len = min(len(ts1), len(ts2))
        ts1 = ts1.iloc[:min_len]
        ts2 = ts2.iloc[:min_len]
    
    # Remove any NaN values
    mask = ~(ts1.isna() | ts2.isna())
    ts1_clean = ts1[mask]
    ts2_clean = ts2[mask]
    
    if len(ts1_clean) < 10:  # Need minimum data points
        return 0.0
    
    # Pearson correlation gives us pattern similarity regardless of scale
    corr, _ = pearsonr(ts1_clean, ts2_clean)
    
    # Convert correlation (-1 to 1) to similarity score (0 to 1)
    similarity = (corr + 1) / 2
    
    return similarity

def perform_clustering(device_ids, normalized_timeseries, parent_label="", depth=0):
    """
    Recursively cluster devices until all classes have <= MAX_DEVICES_PER_CLASS devices.
    
    Args:
        device_ids: List of device IDs to cluster
        normalized_timeseries: Dict of normalized time series for all devices
        parent_label: Label prefix from parent clustering (e.g., "Class_1")
        depth: Current recursion depth
    
    Returns:
        Dict mapping device_id to final class label
    """
    n_devices = len(device_ids)
    
    # Base case: if small enough, don't subdivide
    if n_devices <= MAX_DEVICES_PER_CLASS:
        if parent_label:
            return {dev_id: parent_label for dev_id in device_ids}
        else:
            return {dev_id: "Class_1" for dev_id in device_ids}
    
    # Calculate similarity matrix for this subset
    similarity_matrix = np.zeros((n_devices, n_devices))
    
    for i in range(n_devices):
        for j in range(n_devices):
            if i == j:
                similarity_matrix[i, j] = 1.0
            elif i < j:
                sim = calculate_pattern_similarity(
                    normalized_timeseries[device_ids[i]], 
                    normalized_timeseries[device_ids[j]]
                )
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
    
    # Convert similarity to distance
    distance_matrix = 1 - similarity_matrix
    
    # Determine optimal number of clusters (but ensure we subdivide)
    from sklearn.metrics import silhouette_score
    
    best_silhouette = -1
    optimal_n_clusters = 2  # Default to split in half
    
    # Test different cluster counts
    min_clusters = 2
    max_clusters = min(10, n_devices // 10 + 2)  # Ensure reasonable cluster sizes
    
    for n_clusters in range(min_clusters, max_clusters + 1):
        if n_clusters >= n_devices:
            break
            
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        labels = clustering.fit_predict(distance_matrix)
        
        # Calculate silhouette score
        score = silhouette_score(distance_matrix, labels, metric='precomputed')
        
        if score > best_silhouette:
            best_silhouette = score
            optimal_n_clusters = n_clusters
    
    # Apply clustering
    clustering = AgglomerativeClustering(
        n_clusters=optimal_n_clusters,
        metric='precomputed',
        linkage='average'
    )
    cluster_labels = clustering.fit_predict(distance_matrix)
    
    # Build temporary clusters
    temp_clusters = {}
    for i, dev_id in enumerate(device_ids):
        cluster_id = cluster_labels[i]
        if cluster_id not in temp_clusters:
            temp_clusters[cluster_id] = []
        temp_clusters[cluster_id].append(dev_id)
    
    indent = "  " * depth
    print(f"{indent}🔍 Clustering {n_devices} devices (parent: {parent_label or 'Root'}):")
    print(f"{indent}   ├─ Optimal clusters: {optimal_n_clusters} (Silhouette: {best_silhouette:.3f})")
    
    # Recursively process each cluster
    final_assignments = {}
    
    for cluster_idx, cluster_devices in sorted(temp_clusters.items()):
        # Generate new label
        if parent_label:
            new_label = f"{parent_label}.{cluster_idx + 1}"
        else:
            new_label = f"Class_{cluster_idx + 1}"
        
        cluster_size = len(cluster_devices)
        print(f"{indent}   ├─ {new_label}: {cluster_size} devices", end="")
        
        if cluster_size > MAX_DEVICES_PER_CLASS:
            print(f" (TOO LARGE - subdividing...)")
            # Recursively subdivide
            sub_assignments = perform_clustering(
                cluster_devices, 
                normalized_timeseries, 
                new_label, 
                depth + 1
            )
            final_assignments.update(sub_assignments)
        else:
            print(f" ✓")
            # Assign all devices to this class
            for dev_id in cluster_devices:
                final_assignments[dev_id] = new_label
    
    return final_assignments

def behavioral_pattern_discovery():
    print("=" * 80)
    print("🚀 AHU BEHAVIORAL PATTERN DISCOVERY (RECURSIVE CLUSTERING)")
    print("=" * 80)
    
    # 1. LOAD AND CLEAN ALL 122+ IDs
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: {CSV_PATH} not found.")
        print("💡 Solution: Ensure the CSV file is in the same directory as this script.")
        return

    df_rel = pd.read_csv(CSV_PATH)
    
    # Filter: Keep only rows that HAVE a device_id and ARE NOT marked as "mismatch"
    valid_df = df_rel[df_rel['device_id'].notna() & (df_rel['Remark'] != 'mismatch')]
    all_devices = valid_df['device_id'].unique().tolist()

    print(f"📋 Found {len(all_devices)} unique Device IDs in database.")
    print(f"📅 Extracting 7 days of hourly power data for pattern analysis...")
    print(f"⚙️  Max devices per class: {MAX_DEVICES_PER_CLASS}\n")

    client = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
    query_api = client.query_api()
    
    # Store raw time series for each device
    device_timeseries = {}
    device_metadata = {}
    
    # 2. EXTRACT 1 WEEK OF DATA (168 hourly points for pattern analysis)
    print("Phase 1: Data Extraction")
    print("-" * 80)
    
    for idx, dev_id in enumerate(all_devices, 1):
        measurement = f"wach_{dev_id}_power_total"
        query = f'''
        from(bucket: "{BUCKET}")
          |> range(start: -7d)
          |> filter(fn: (r) => r["_measurement"] == "{measurement}")
          |> filter(fn: (r) => r["_field"] == "value")
          |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
          |> sort(columns: ["_time"])
        '''
        try:
            df = query_api.query_data_frame(query)
            if df is not None and not df.empty and len(df) > 24:  # At least 1 day of data
                # Store raw time series
                df['_time'] = pd.to_datetime(df['_time'])
                df = df.set_index('_time').sort_index()
                
                device_timeseries[dev_id] = df['_value']
                
                # Store metadata
                meta = valid_df[valid_df['device_id'] == dev_id].iloc[0]
                device_metadata[dev_id] = {
                    'dept': meta['Department Name'],
                    'area': meta['Area Name'],
                    'label': meta['AHU Label'],
                    'type': meta['Type']
                }
                
                print(f"  [{idx:3d}/{len(all_devices)}] ✓ {dev_id}: {len(df)} hours ({meta['Area Name'][:50]})")
            else:
                print(f"  [{idx:3d}/{len(all_devices)}] ✗ {dev_id}: Insufficient data")
        except Exception as e:
            print(f"  [{idx:3d}/{len(all_devices)}] ✗ {dev_id}: Error - {str(e)[:50]}")
            continue

    if not device_timeseries:
        print("\n❌ No data found in InfluxDB for the provided IDs.")
        client.close()
        return

    print(f"\n✅ Successfully extracted data for {len(device_timeseries)} devices")
    
    # 3. PLOT ALL TIME SERIES ON ONE GRAPH
    print("\n" + "=" * 80)
    print("Phase 2: Visualizing Raw Time Series")
    print("-" * 80)
    
    plt.figure(figsize=(24, 12))
    colors = plt.cm.tab20(np.linspace(0, 1, len(device_timeseries)))
    
    for idx, (dev_id, ts) in enumerate(device_timeseries.items()):
        plt.plot(ts.index, ts.values, alpha=0.3, linewidth=0.8, color=colors[idx], label=dev_id)
    
    plt.xlabel('Time', fontsize=14, fontweight='bold')
    plt.ylabel('Power (kW)', fontsize=14, fontweight='bold')
    plt.title(f'7-Day Power Consumption - All {len(device_timeseries)} AHU Devices', 
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    raw_plot_file = os.path.join(BASE_DIR, "all_devices_7day_raw.png")
    plt.savefig(raw_plot_file, dpi=150, bbox_inches='tight')
    print(f"📊 Raw time series plot saved: {raw_plot_file}")
    plt.close()
    
    # 4. NORMALIZE ALL TIME SERIES (for pattern comparison)
    print("\n" + "=" * 80)
    print("Phase 3: Normalizing Time Series for Pattern Analysis")
    print("-" * 80)
    
    normalized_timeseries = {}
    for dev_id, ts in device_timeseries.items():
        normalized_timeseries[dev_id] = normalize_timeseries(ts)
    
    # Plot normalized patterns
    plt.figure(figsize=(24, 12))
    for idx, (dev_id, ts) in enumerate(normalized_timeseries.items()):
        plt.plot(ts.index, ts.values, alpha=0.3, linewidth=0.8, color=colors[idx], label=dev_id)
    
    plt.xlabel('Time', fontsize=14, fontweight='bold')
    plt.ylabel('Normalized Power (0-1)', fontsize=14, fontweight='bold')
    plt.title(f'7-Day Normalized Patterns - All {len(device_timeseries)} AHU Devices', 
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    norm_plot_file = os.path.join(BASE_DIR, "all_devices_7day_normalized.png")
    plt.savefig(norm_plot_file, dpi=150, bbox_inches='tight')
    print(f"📊 Normalized pattern plot saved: {norm_plot_file}")
    plt.close()
    
    # 5. CALCULATE PAIRWISE PATTERN SIMILARITY MATRIX (for full dataset)
    print("\n" + "=" * 80)
    print("Phase 4: Calculating Pattern Similarity Matrix")
    print("-" * 80)
    
    device_ids = list(normalized_timeseries.keys())
    n_devices = len(device_ids)
    
    # Initialize similarity matrix
    similarity_matrix = np.zeros((n_devices, n_devices))
    
    print(f"Computing {n_devices}x{n_devices} = {n_devices**2} pairwise similarities...")
    
    for i in range(n_devices):
        for j in range(n_devices):
            if i == j:
                similarity_matrix[i, j] = 1.0
            elif i < j:  # Only compute upper triangle
                sim = calculate_pattern_similarity(
                    normalized_timeseries[device_ids[i]], 
                    normalized_timeseries[device_ids[j]]
                )
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim  # Mirror to lower triangle
    
    print("✅ Similarity matrix computed")
    
    # 6. CREATE SIMILARITY HEATMAP
    print("\n" + "=" * 80)
    print("Phase 5: Generating Pattern Similarity Heatmap")
    print("-" * 80)
    
    plt.figure(figsize=(28, 24))
    
    # Create a mask for better visualization
    mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
    
    # Create heatmap with annotations for high similarity
    ax = sns.heatmap(
        similarity_matrix, 
        xticklabels=device_ids,
        yticklabels=device_ids,
        cmap="RdYlGn",
        vmin=0, vmax=1,
        center=0.7,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pattern Similarity (0=Different, 1=Identical)"},
        mask=mask
    )
    
    plt.title(f'AHU Behavioral Pattern Similarity Matrix ({n_devices} Devices)\n' + 
              'Based on 7-Day Normalized Power Consumption Patterns',
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Device ID', fontsize=14, fontweight='bold')
    plt.ylabel('Device ID', fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    heatmap_file = os.path.join(BASE_DIR, "pattern_similarity_heatmap.png")
    plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
    print(f"🔥 Pattern similarity heatmap saved: {heatmap_file}")
    plt.close()
    
    # 7. RECURSIVE HIERARCHICAL CLUSTERING
    print("\n" + "=" * 80)
    print("Phase 6: Recursive Clustering (Max {MAX_DEVICES_PER_CLASS} devices/class)")
    print("=" * 80)
    print()
    
    # Perform recursive clustering
    device_class_assignments = perform_clustering(device_ids, normalized_timeseries)
    
    # 8. ANALYZE AND CATEGORIZE BEHAVIORAL CLASSES
    print("\n" + "=" * 80)
    print("Phase 7: Behavioral Class Analysis")
    print("=" * 80)
    
    # Group devices by final class
    behavioral_classes = {}
    for dev_id, class_label in device_class_assignments.items():
        if class_label not in behavioral_classes:
            behavioral_classes[class_label] = []
        behavioral_classes[class_label].append(dev_id)
    
    # Calculate statistics for each device
    stats_list = []
    
    for dev_id in device_ids:
        class_label = device_class_assignments[dev_id]
        dev_idx = device_ids.index(dev_id)
        
        # Calculate statistics
        ts_raw = device_timeseries[dev_id]
        
        # Find devices most similar to this one
        similarities = similarity_matrix[dev_idx]
        top_similar_idx = np.argsort(similarities)[-6:-1][::-1]  # Top 5 (excluding itself)
        top_similar = [device_ids[idx] for idx in top_similar_idx]
        top_similar_scores = [similarities[idx] for idx in top_similar_idx]
        
        # Calculate average similarity to devices in the same class
        class_devices = behavioral_classes[class_label]
        class_indices = [device_ids.index(d) for d in class_devices]
        class_similarities = [similarities[idx] for idx in class_indices if idx != dev_idx]
        avg_class_sim = np.mean(class_similarities) if class_similarities else 1.0
        
        stats_list.append({
            'device_id': dev_id,
            'behavioral_class': class_label,
            'dept': device_metadata[dev_id]['dept'],
            'area': device_metadata[dev_id]['area'],
            'label': device_metadata[dev_id]['label'],
            'type': device_metadata[dev_id]['type'],
            'avg_kw': round(ts_raw.mean(), 2),
            'peak_kw': round(ts_raw.max(), 2),
            'min_kw': round(ts_raw.min(), 2),
            'std_kw': round(ts_raw.std(), 2),
            'data_points': len(ts_raw),
            'avg_similarity_to_class': round(avg_class_sim, 3),
            'top_5_similar_devices': ', '.join(top_similar),
            'top_5_similarity_scores': ', '.join([f'{s:.3f}' for s in top_similar_scores])
        })
    
    # Print class summaries
    print(f"\n📊 Found {len(behavioral_classes)} Final Behavioral Classes:\n")
    
    for class_label in sorted(behavioral_classes.keys(), key=lambda x: (len(x.split('.')), x)):
        devices = behavioral_classes[class_label]
        
        # Calculate average intra-class similarity
        class_indices = [device_ids.index(d) for d in devices]
        intra_class_sims = []
        for i in class_indices:
            for j in class_indices:
                if i < j:
                    intra_class_sims.append(similarity_matrix[i, j])
        
        avg_intra_sim = np.mean(intra_class_sims) if intra_class_sims else 0
        
        indent = "  " * (len(class_label.split('.')) - 1)
        print(f"{indent}🏷️  {class_label} ({len(devices)} devices)")
        print(f"{indent}   └─ Avg Intra-Class Similarity: {avg_intra_sim:.1%}")
        print(f"{indent}   └─ Devices: {', '.join(devices[:10])}")
        if len(devices) > 10:
            print(f"{indent}               ... and {len(devices) - 10} more")
        print()
    
    # 9. SAVE DETAILED CSV REPORT
    print("=" * 80)
    print("Phase 8: Generating Reports")
    print("-" * 80)
    
    summary_df = pd.DataFrame(stats_list)
    summary_df = summary_df.sort_values(['behavioral_class', 'avg_similarity_to_class'], ascending=[True, False])
    
    csv_file = os.path.join(BASE_DIR, "behavioral_pattern_analysis.csv")
    summary_df.to_csv(csv_file, index=False)
    print(f"📄 Detailed analysis saved: {csv_file}")
    
    # 10. CREATE CLASS-SPECIFIC VISUALIZATIONS
    print("\nGenerating class-specific visualizations...")
    
    for class_label in sorted(behavioral_classes.keys(), key=lambda x: (len(x.split('.')), x)):
        devices = behavioral_classes[class_label]
        safe_class_name = class_label.replace('.', '_').lower()
        
        # Plot all devices in this class
        plt.figure(figsize=(20, 10))
        
        # Raw data subplot
        plt.subplot(2, 1, 1)
        for dev_id in devices:
            ts = device_timeseries[dev_id]
            plt.plot(ts.index, ts.values, alpha=0.6, linewidth=1.5, label=dev_id)
        plt.ylabel('Power (kW)', fontsize=12, fontweight='bold')
        plt.title(f'{class_label} - Raw Power Consumption ({len(devices)} devices)', 
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Normalized pattern subplot
        plt.subplot(2, 1, 2)
        for dev_id in devices:
            ts = normalized_timeseries[dev_id]
            plt.plot(ts.index, ts.values, alpha=0.6, linewidth=1.5, label=dev_id)
        plt.xlabel('Time', fontsize=12, fontweight='bold')
        plt.ylabel('Normalized Power (0-1)', fontsize=12, fontweight='bold')
        plt.title(f'{class_label} - Normalized Behavioral Pattern', 
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        class_plot_file = os.path.join(BASE_DIR, f"behavioral_{safe_class_name}_patterns.png")
        plt.savefig(class_plot_file, dpi=150, bbox_inches='tight')
        print(f"  ✓ {class_label} visualization: {class_plot_file}")
        plt.close()
    
    # 11. CREATE SIMILARITY DISTRIBUTION PLOT
    print("\nGenerating similarity distribution analysis...")
    
    plt.figure(figsize=(16, 6))
    
    # Flatten upper triangle of similarity matrix (excluding diagonal)
    upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
    all_similarities = similarity_matrix[upper_triangle_indices]
    
    plt.subplot(1, 2, 1)
    plt.hist(all_similarities, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    plt.xlabel('Pattern Similarity Score', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title('Distribution of Pairwise Pattern Similarities', fontsize=14, fontweight='bold')
    plt.axvline(0.9, color='red', linestyle='--', linewidth=2, label='90% threshold')
    plt.axvline(0.8, color='orange', linestyle='--', linewidth=2, label='80% threshold')
    plt.axvline(0.7, color='yellow', linestyle='--', linewidth=2, label='70% threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    thresholds = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]
    counts = [np.sum(all_similarities >= t) for t in thresholds]
    plt.barh(thresholds, counts, color='steelblue', alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Device Pairs', fontsize=12, fontweight='bold')
    plt.ylabel('Similarity Threshold', fontsize=12, fontweight='bold')
    plt.title('Device Pairs Above Each Similarity Threshold', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    for i, (t, c) in enumerate(zip(thresholds, counts)):
        plt.text(c + max(counts)*0.01, t, f'{c}', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    dist_plot_file = os.path.join(BASE_DIR, "similarity_distribution.png")
    plt.savefig(dist_plot_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Similarity distribution: {dist_plot_file}")
    plt.close()
    
    # 12. FINAL SUMMARY STATISTICS
    print("\n" + "=" * 80)
    print("📊 FINAL SUMMARY STATISTICS")
    print("=" * 80)
    
    print(f"\n✅ Total Devices Analyzed: {n_devices}")
    print(f"✅ Behavioral Classes Identified: {len(behavioral_classes)}")
    print(f"✅ Largest Class Size: {max(len(devices) for devices in behavioral_classes.values())} devices")
    print(f"✅ Smallest Class Size: {min(len(devices) for devices in behavioral_classes.values())} devices")
    print(f"✅ Average Pattern Similarity: {all_similarities.mean():.1%}")
    print(f"✅ Median Pattern Similarity: {np.median(all_similarities):.1%}")
    print(f"\n📈 Similarity Thresholds:")
    print(f"   • ≥90% (Very Strong): {np.sum(all_similarities >= 0.90)} device pairs ({np.sum(all_similarities >= 0.90)/len(all_similarities)*100:.1f}%)")
    print(f"   • ≥80% (Strong): {np.sum(all_similarities >= 0.80)} device pairs ({np.sum(all_similarities >= 0.80)/len(all_similarities)*100:.1f}%)")
    print(f"   • ≥70% (Moderate): {np.sum(all_similarities >= 0.70)} device pairs ({np.sum(all_similarities >= 0.70)/len(all_similarities)*100:.1f}%)")
    print(f"   • ≥60% (Weak): {np.sum(all_similarities >= 0.60)} device pairs ({np.sum(all_similarities >= 0.60)/len(all_similarities)*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("✅ BEHAVIORAL PATTERN DISCOVERY COMPLETE")
    print("=" * 80)
    
    print("\n📁 Generated Files:")
    print(f"   1. {raw_plot_file}")
    print(f"   2. {norm_plot_file}")
    print(f"   3. {heatmap_file}")
    print(f"   4. {csv_file}")
    print(f"   5. {dist_plot_file}")
    for idx, class_label in enumerate(sorted(behavioral_classes.keys(), key=lambda x: (len(x.split('.')), x)), 6):
        safe_name = class_label.replace('.', '_').lower()
        print(f"   {idx}. behavioral_{safe_name}_patterns.png")
    
    print("\n💡 Next Steps:")
    print("   1. Review the behavioral_pattern_analysis.csv for detailed device classifications")
    print("   2. Examine the pattern_similarity_heatmap.png to identify strong relationships")
    print("   3. Review individual class visualizations to understand each behavioral group")
    print("   4. Use the 'top_5_similar_devices' column to find devices with matching patterns")
    print(f"   5. Note: All classes now contain ≤{MAX_DEVICES_PER_CLASS} devices for easier management")
    
    client.close()

if __name__ == "__main__":
    behavioral_pattern_discovery()