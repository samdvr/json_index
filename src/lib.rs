use anyhow::{Context, Result};
use fjall::{PartitionCreateOptions, PartitionHandle};
use serde_json::Value;
use std::collections::HashSet;

/// Represents the logical operation to apply in complex queries
#[derive(Debug, Clone, Copy)]
pub enum QueryLogic {
    /// All conditions must match (intersection of results)
    And,
    /// Any condition can match (union of results)
    Or,
}

/// Main database structure that manages indexing and querying
pub struct JsonIndexDB {
    index_partition: PartitionHandle,
    doc_partition: PartitionHandle,
}

impl JsonIndexDB {
    /// Create a new jsonindexDB instance
    pub fn new(keyspace: fjall::Keyspace) -> Result<Self> {
        // Each partition is its own physical LSM-tree
        let index_partition =
            keyspace.open_partition("index_partition", PartitionCreateOptions::default())?;

        let doc_partition =
            keyspace.open_partition("doc_partition", PartitionCreateOptions::default())?;

        Ok(Self {
            index_partition,
            doc_partition,
        })
    }

    /// Index a document with the given document ID
    pub fn index_document(&self, doc: &str, doc_id: &str) -> Result<()> {
        let json_val: Value =
            serde_json::from_str(doc).context("Failed to parse document as JSON")?;

        // Store the original document
        self.doc_partition.insert(doc_id, doc)?;

        // Flatten the JSON structure for indexing
        let flattened = match json_val {
            Value::Object(ref map) => {
                let flat_entries = flatten_serde_json::flatten(map);
                flat_entries
                    .into_iter()
                    .map(|(k, v)| {
                        let value_str = match v {
                            Value::String(s) => s,
                            _ => v.to_string(),
                        };
                        format!("{}={}", k, value_str)
                    })
                    .collect::<Vec<String>>()
            }
            _ => {
                return Err(anyhow::anyhow!("Root JSON value must be an object"));
            }
        };

        // Update the index with flattened key-value pairs
        for key in flattened {
            self.add_to_index(&key, doc_id)?;
        }

        Ok(())
    }

    /// Add a document ID to an index entry
    fn add_to_index(&self, key: &str, doc_id: &str) -> Result<()> {
        // Check if the key already exists in the index
        let existing = self.index_partition.get(key)?;

        let doc_ids = match existing {
            Some(ids) => {
                // Deserialize existing array of doc_ids
                let mut ids: Vec<String> = serde_json::from_slice(&ids)?;

                // Only add the doc_id if it's not already in the array
                if !ids.contains(&doc_id.to_string()) {
                    ids.push(doc_id.to_string());
                }
                ids
            }
            None => {
                // Create a new array with just this doc_id
                vec![doc_id.to_string()]
            }
        };

        // Serialize and store the updated array
        let serialized = serde_json::to_vec(&doc_ids)?;
        self.index_partition.insert(key, serialized)?;

        Ok(())
    }

    /// Query documents that match the given key-value pattern
    pub fn query(&self, query: &str) -> Result<Vec<(String, String)>> {
        // Get document IDs that match the query
        let doc_ids = self.get_matching_doc_ids(query)?;

        // Retrieve the actual documents
        self.get_documents(&doc_ids)
    }

    /// Complex query with multiple conditions combined with AND/OR logic
    pub fn complex_query(
        &self,
        queries: &[&str],
        logic: QueryLogic,
    ) -> Result<Vec<(String, String)>> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }

        // Get the first set of results
        let mut result_doc_ids = self.get_matching_doc_ids(queries[0])?;
        let mut result_set: HashSet<String> = result_doc_ids.into_iter().collect();

        // Process subsequent queries based on the logic
        for query in &queries[1..] {
            let new_doc_ids = self.get_matching_doc_ids(query)?;
            let new_set: HashSet<String> = new_doc_ids.into_iter().collect();

            match logic {
                QueryLogic::And => {
                    // Intersection - keep only IDs that appear in both sets
                    result_set = result_set.intersection(&new_set).cloned().collect();
                }
                QueryLogic::Or => {
                    // Union - keep IDs that appear in either set
                    result_set = result_set.union(&new_set).cloned().collect();
                }
            }
        }

        // Convert back to vector and retrieve the documents
        result_doc_ids = result_set.into_iter().collect();
        self.get_documents(&result_doc_ids)
    }

    /// Get document IDs that match a specific query
    fn get_matching_doc_ids(&self, query: &str) -> Result<Vec<String>> {
        match self.index_partition.get(query)? {
            Some(ids_bytes) => {
                let doc_ids: Vec<String> = serde_json::from_slice(&ids_bytes)?;
                Ok(doc_ids)
            }
            None => Ok(Vec::new()),
        }
    }

    /// Retrieve documents by their IDs
    fn get_documents(&self, doc_ids: &[String]) -> Result<Vec<(String, String)>> {
        let mut results = Vec::with_capacity(doc_ids.len());

        for doc_id in doc_ids {
            if let Some(doc_bytes) = self.doc_partition.get(doc_id)? {
                let doc_str = String::from_utf8(doc_bytes.to_vec())?;
                results.push((doc_id.clone(), doc_str));
            }
        }

        Ok(results)
    }

    /// Delete a document and its index entries
    pub fn delete_document(&self, doc_id: &str) -> Result<()> {
        // First get the document
        let doc_bytes = match self.doc_partition.get(doc_id)? {
            Some(bytes) => bytes,
            None => return Ok(()), // Document doesn't exist, nothing to delete
        };

        // Parse the document to recreate the index keys
        let doc_str = String::from_utf8(doc_bytes.to_vec())?;
        let json_val: Value = serde_json::from_str(&doc_str)?;

        // Flatten the JSON structure to get all the keys we need to update
        let flattened = match json_val {
            Value::Object(ref map) => {
                let flat_entries = flatten_serde_json::flatten(map);
                flat_entries
                    .into_iter()
                    .map(|(k, v)| {
                        let value_str = match v {
                            Value::String(s) => s,
                            _ => v.to_string(),
                        };
                        format!("{}={}", k, value_str)
                    })
                    .collect::<Vec<String>>()
            }
            _ => {
                return Err(anyhow::anyhow!("Root JSON value must be an object"));
            }
        };

        // Remove this document ID from all relevant index entries
        for key in flattened {
            self.remove_from_index(&key, doc_id)?;
        }

        // Finally, delete the document itself
        self.doc_partition.remove(doc_id)?;

        Ok(())
    }

    /// Remove a document ID from an index entry
    fn remove_from_index(&self, key: &str, doc_id: &str) -> Result<()> {
        if let Some(ids_bytes) = self.index_partition.get(key)? {
            let mut doc_ids: Vec<String> = serde_json::from_slice(&ids_bytes)?;

            // Remove the doc_id if it exists
            doc_ids.retain(|id| id != doc_id);

            if doc_ids.is_empty() {
                // If no documents left, remove the index entry entirely
                self.index_partition.remove(key)?;
            } else {
                // Update the index with the remaining document IDs
                let serialized = serde_json::to_vec(&doc_ids)?;
                self.index_partition.insert(key, serialized)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use fjall::Config;

    use super::*;
    use std::fs;
    use std::path::Path;

    #[test]
    fn test_basic_flow() -> Result<()> {
        // Create a temporary directory for testing
        let test_dir = "test_db_data";
        if Path::new(test_dir).exists() {
            fs::remove_dir_all(test_dir)?;
        }

        // Create a database instance
        let keyspace = Config::new(test_dir).open()?;
        let jsonindex = JsonIndexDB::new(keyspace)?;

        // Test document
        let doc = r#"{
            "name": "Test User",
            "age": 25,
            "address": {
                "city": "Test City",
                "country": "Test Country"
            }
        }"#;

        // Index the document
        let doc_id = "test-doc-123";
        jsonindex.index_document(doc, doc_id)?;

        // Test simple query
        let results = jsonindex.query("name=Test User")?;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, doc_id);

        // Test complex AND query - both conditions match
        let results = jsonindex.complex_query(&["name=Test User", "age=25"], QueryLogic::And)?;
        assert_eq!(results.len(), 1);

        // Test complex AND query - one condition doesn't match
        let results = jsonindex.complex_query(&["name=Test User", "age=30"], QueryLogic::And)?;
        assert_eq!(results.len(), 0);

        // Test complex OR query
        let results = jsonindex.complex_query(&["age=25", "age=30"], QueryLogic::Or)?;
        assert_eq!(results.len(), 1);

        // Test nested field query
        let results = jsonindex.query("address.city=Test City")?;
        assert_eq!(results.len(), 1);

        // Test deletion
        jsonindex.delete_document(doc_id)?;
        let results = jsonindex.query("name=Test User")?;
        assert_eq!(results.len(), 0);

        // Clean up test directory
        fs::remove_dir_all(test_dir)?;

        Ok(())
    }
}
