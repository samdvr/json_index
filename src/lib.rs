use anyhow::{Context, Result};
use fjall::{PartitionCreateOptions, PartitionHandle};
use serde_json::Value;
use std::collections::HashSet;
use std::convert::TryFrom;
use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

/// Represents the logical operation to apply in complex queries
#[derive(Debug, Clone, Copy)]
pub enum QueryLogic {
    /// All conditions must match (intersection of results)
    And,
    /// Any condition can match (union of results)
    Or,
}

/// Supported comparison operators for queries
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ComparisonOperator {
    /// Exact equality (=)
    Equals,
    /// Contains substring
    Contains,
    /// Greater than (>)
    GreaterThan,
    /// Less than (<)
    LessThan,
}

impl fmt::Display for ComparisonOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComparisonOperator::Equals => write!(f, "="),
            ComparisonOperator::Contains => write!(f, "contains"),
            ComparisonOperator::GreaterThan => write!(f, ">"),
            ComparisonOperator::LessThan => write!(f, "<"),
        }
    }
}

/// Represents a parsed query condition
#[derive(Debug, Clone)]
pub struct QueryCondition {
    /// Field path (can be nested with dot notation)
    pub field: String,
    /// Operator for comparison
    pub operator: ComparisonOperator,
    /// Value to compare against
    pub value: String,
}

impl QueryCondition {
    pub fn new(field: &str, operator: ComparisonOperator, value: &str) -> Self {
        Self {
            field: field.to_string(),
            operator,
            value: value.to_string(),
        }
    }

    /// Convert the condition to an index key for storage/retrieval
    pub fn to_index_key(&self) -> String {
        // For now, only equals is directly indexed
        if self.operator == ComparisonOperator::Equals {
            format!("{}={}", self.field, self.value)
        } else {
            // For other operators, we'll need to do filtering later
            // Just include the field in the key so we can retrieve all values
            format!("{}:{}", self.field, self.operator)
        }
    }
}

impl TryFrom<&str> for QueryCondition {
    type Error = anyhow::Error;

    fn try_from(query_str: &str) -> Result<Self, Self::Error> {
        // Check for contains operator
        if let Some(pos) = query_str.find(" contains ") {
            let field = &query_str[0..pos];
            let value = &query_str[pos + 10..];
            return Ok(QueryCondition::new(
                field,
                ComparisonOperator::Contains,
                value,
            ));
        }

        // Check for greater than
        if let Some(pos) = query_str.find('>') {
            let field = &query_str[0..pos];
            let value = &query_str[pos + 1..];
            return Ok(QueryCondition::new(
                field,
                ComparisonOperator::GreaterThan,
                value,
            ));
        }

        // Check for less than
        if let Some(pos) = query_str.find('<') {
            let field = &query_str[0..pos];
            let value = &query_str[pos + 1..];
            return Ok(QueryCondition::new(
                field,
                ComparisonOperator::LessThan,
                value,
            ));
        }

        // Default to equals
        if let Some(pos) = query_str.find('=') {
            let field = &query_str[0..pos];
            let value = &query_str[pos + 1..];
            return Ok(QueryCondition::new(
                field,
                ComparisonOperator::Equals,
                value,
            ));
        }

        Err(anyhow::anyhow!("Invalid query format: {}", query_str))
    }
}

/// Custom error types for JsonIndexDB operations
#[derive(Debug)]
pub enum IndexError {
    /// Error parsing or processing JSON
    JsonError(String),
    /// Error accessing the document store
    DocumentStoreError(String),
    /// Error accessing the index store
    IndexStoreError(String),
    /// Invalid query format or structure
    QueryError(String),
    /// Invalid document ID
    DocumentIdError(String),
    /// Database operation failed
    OperationError(String),
    /// Other error with context
    Other(String),
}

impl fmt::Display for IndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IndexError::JsonError(msg) => write!(f, "JSON error: {}", msg),
            IndexError::DocumentStoreError(msg) => write!(f, "Document store error: {}", msg),
            IndexError::IndexStoreError(msg) => write!(f, "Index store error: {}", msg),
            IndexError::QueryError(msg) => write!(f, "Query error: {}", msg),
            IndexError::DocumentIdError(msg) => write!(f, "Document ID error: {}", msg),
            IndexError::OperationError(msg) => write!(f, "Operation error: {}", msg),
            IndexError::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl Error for IndexError {}

impl From<serde_json::Error> for IndexError {
    fn from(err: serde_json::Error) -> Self {
        IndexError::JsonError(err.to_string())
    }
}

impl From<std::string::FromUtf8Error> for IndexError {
    fn from(err: std::string::FromUtf8Error) -> Self {
        IndexError::JsonError(format!("UTF-8 conversion error: {}", err))
    }
}

/// A strongly typed document ID for better type safety
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DocumentId(String);

impl DocumentId {
    /// Create a new document ID
    pub fn new(id: &str) -> Result<Self, IndexError> {
        // Validate document ID format
        if id.is_empty() {
            return Err(IndexError::DocumentIdError(
                "Document ID cannot be empty".to_string(),
            ));
        }

        // Add more validation as needed
        // For example, check for valid characters, length limits, etc.

        Ok(DocumentId(id.to_string()))
    }

    /// Get the underlying string representation
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for DocumentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<DocumentId> for String {
    fn from(id: DocumentId) -> Self {
        id.0
    }
}

/// Configuration options for JsonIndexDB
#[derive(Debug, Clone)]
pub struct IndexConfig {
    /// Case sensitivity for string comparisons (default: true)
    pub case_sensitive: bool,
    /// Whether to normalize strings before indexing (default: false)
    pub normalize_strings: bool,
    /// Maximum number of results to return from a query (default: 1000)
    pub max_results: usize,
    /// Default partitions names
    pub index_partition_name: String,
    pub doc_partition_name: String,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            case_sensitive: true,
            normalize_strings: false,
            max_results: 1000,
            index_partition_name: "index_partition".to_string(),
            doc_partition_name: "doc_partition".to_string(),
        }
    }
}

/// Transaction state for atomic operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransactionState {
    /// Transaction is active
    Active,
    /// Transaction has been committed
    Committed,
    /// Transaction has been rolled back
    RolledBack,
}

/// A transaction that allows for atomic operations
pub struct Transaction {
    /// Unique identifier for this transaction
    id: String,
    /// Current state of the transaction
    state: TransactionState,
    /// Operations performed in this transaction
    operations: Vec<TransactionOperation>,
}

/// Operations that can be performed in a transaction
#[derive(Debug, Clone)]
enum TransactionOperation {
    /// Index a document
    IndexDocument { doc: String, doc_id: String },
    /// Delete a document
    DeleteDocument { doc_id: String },
}

impl Transaction {
    /// Create a new transaction
    fn new() -> Self {
        let transaction_id = format!(
            "txn-{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or(Duration::from_secs(0))
                .as_nanos()
        );

        Self {
            id: transaction_id,
            state: TransactionState::Active,
            operations: Vec::new(),
        }
    }

    /// Add a document indexing operation to the transaction
    pub fn index_document(&mut self, doc: &str, doc_id: &str) -> Result<(), IndexError> {
        if self.state != TransactionState::Active {
            return Err(IndexError::OperationError(format!(
                "Cannot add operation to {} transaction",
                self.state
            )));
        }

        self.operations.push(TransactionOperation::IndexDocument {
            doc: doc.to_string(),
            doc_id: doc_id.to_string(),
        });

        Ok(())
    }

    /// Add a document deletion operation to the transaction
    pub fn delete_document(&mut self, doc_id: &str) -> Result<(), IndexError> {
        if self.state != TransactionState::Active {
            return Err(IndexError::OperationError(format!(
                "Cannot add operation to {} transaction",
                self.state
            )));
        }

        self.operations.push(TransactionOperation::DeleteDocument {
            doc_id: doc_id.to_string(),
        });

        Ok(())
    }

    /// Get the unique ID of this transaction
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get the current state of this transaction
    pub fn state(&self) -> TransactionState {
        self.state
    }

    /// Mark this transaction as committed
    fn commit(&mut self) {
        self.state = TransactionState::Committed;
    }

    /// Mark this transaction as rolled back
    fn rollback(&mut self) {
        self.state = TransactionState::RolledBack;
    }
}

impl fmt::Display for TransactionState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransactionState::Active => write!(f, "active"),
            TransactionState::Committed => write!(f, "committed"),
            TransactionState::RolledBack => write!(f, "rolled back"),
        }
    }
}

/// Thread-safe database wrapper with concurrency control
pub struct ThreadSafeJsonIndexDB {
    /// The underlying database instance
    db: Arc<Mutex<JsonIndexDB>>,
    /// Configuration options
    config: IndexConfig,
}

impl ThreadSafeJsonIndexDB {
    /// Create a new thread-safe JsonIndexDB instance
    pub fn new(keyspace: fjall::Keyspace) -> Result<Self> {
        let db = JsonIndexDB::new(keyspace)?;
        Ok(Self {
            db: Arc::new(Mutex::new(db)),
            config: IndexConfig::default(),
        })
    }

    /// Create a new thread-safe JsonIndexDB instance with custom configuration
    pub fn with_config(keyspace: fjall::Keyspace, config: IndexConfig) -> Result<Self> {
        let db = JsonIndexDB::with_config(keyspace, config.clone())?;
        Ok(Self {
            db: Arc::new(Mutex::new(db)),
            config,
        })
    }

    /// Start a new transaction
    pub fn begin_transaction(&self) -> Transaction {
        Transaction::new()
    }

    /// Commit a transaction by executing all its operations atomically
    pub async fn commit_transaction(&self, mut transaction: Transaction) -> Result<(), IndexError> {
        if transaction.state() != TransactionState::Active {
            return Err(IndexError::OperationError(format!(
                "Cannot commit {} transaction",
                transaction.state()
            )));
        }

        // Execute all operations, each gets its own lock to avoid holding across awaits
        for operation in &transaction.operations {
            match operation {
                TransactionOperation::IndexDocument { doc, doc_id } => {
                    self.index_document(doc, doc_id).await?;
                }
                TransactionOperation::DeleteDocument { doc_id } => {
                    self.delete_document(doc_id).await?;
                }
            }
        }

        // Mark transaction as committed
        transaction.commit();

        Ok(())
    }

    /// Roll back a transaction without executing its operations
    pub fn rollback_transaction(&self, mut transaction: Transaction) -> Result<(), IndexError> {
        if transaction.state() != TransactionState::Active {
            return Err(IndexError::OperationError(format!(
                "Cannot rollback {} transaction",
                transaction.state()
            )));
        }

        // Mark transaction as rolled back
        transaction.rollback();

        Ok(())
    }

    /// Index a document with thread-safety
    pub async fn index_document(&self, doc: &str, doc_id: &str) -> Result<(), IndexError> {
        // Clone values to use across await boundaries
        let doc = doc.to_string();
        let doc_id = doc_id.to_string();

        // Get a copy of the database for the operation
        let db = match self.db.lock() {
            Ok(db) => Arc::new(db.clone()),
            Err(_) => {
                return Err(IndexError::OperationError(
                    "Failed to acquire mutex".to_string(),
                ));
            }
        };

        // Drop the lock before awaiting
        tokio::task::spawn_blocking(move || {
            let rt = tokio::runtime::Handle::current();
            rt.block_on(db.index_document(&doc, &doc_id))
        })
        .await
        .map_err(|e| IndexError::OperationError(format!("Task join error: {}", e)))?
        .map_err(|e| IndexError::OperationError(e.to_string()))
    }

    /// Delete a document with thread-safety
    pub async fn delete_document(&self, doc_id: &str) -> Result<(), IndexError> {
        // Clone value to use across await boundaries
        let doc_id = doc_id.to_string();

        // Get a copy of the database for the operation
        let db = match self.db.lock() {
            Ok(db) => Arc::new(db.clone()),
            Err(_) => {
                return Err(IndexError::OperationError(
                    "Failed to acquire mutex".to_string(),
                ));
            }
        };

        // Drop the lock before awaiting
        tokio::task::spawn_blocking(move || {
            let rt = tokio::runtime::Handle::current();
            rt.block_on(db.delete_document(&doc_id))
        })
        .await
        .map_err(|e| IndexError::OperationError(format!("Task join error: {}", e)))?
        .map_err(|e| IndexError::OperationError(e.to_string()))
    }

    /// Query documents with thread-safety
    pub async fn query(&self, query: &str) -> Result<Vec<(String, String)>, IndexError> {
        // Clone value to use across await boundaries
        let query = query.to_string();

        // Get a copy of the database for the operation
        let db = match self.db.lock() {
            Ok(db) => Arc::new(db.clone()),
            Err(_) => {
                return Err(IndexError::OperationError(
                    "Failed to acquire mutex".to_string(),
                ));
            }
        };

        // Drop the lock before awaiting
        tokio::task::spawn_blocking(move || {
            let rt = tokio::runtime::Handle::current();
            rt.block_on(db.query(&query))
        })
        .await
        .map_err(|e| IndexError::OperationError(format!("Task join error: {}", e)))?
        .map_err(|e| IndexError::OperationError(e.to_string()))
    }

    /// Complex query with thread-safety
    pub async fn complex_query(
        &self,
        queries: &[&str],
        logic: QueryLogic,
    ) -> Result<Vec<(String, String)>, IndexError> {
        // Clone values to use across await boundaries
        let queries: Vec<String> = queries.iter().map(|q| q.to_string()).collect();

        // Get a copy of the database for the operation
        let db = match self.db.lock() {
            Ok(db) => Arc::new(db.clone()),
            Err(_) => {
                return Err(IndexError::OperationError(
                    "Failed to acquire mutex".to_string(),
                ));
            }
        };

        // Drop the lock before awaiting
        tokio::task::spawn_blocking(move || {
            let rt = tokio::runtime::Handle::current();
            let query_refs: Vec<&str> = queries.iter().map(|q| q.as_str()).collect();
            rt.block_on(db.complex_query(&query_refs, logic))
        })
        .await
        .map_err(|e| IndexError::OperationError(format!("Task join error: {}", e)))?
        .map_err(|e| IndexError::OperationError(e.to_string()))
    }

    /// Query with condition using thread-safety
    pub async fn query_with_condition(
        &self,
        condition: &QueryCondition,
    ) -> Result<Vec<(String, String)>, IndexError> {
        // Clone value to use across await boundaries
        let condition = condition.clone();

        // Get a copy of the database for the operation
        let db = match self.db.lock() {
            Ok(db) => Arc::new(db.clone()),
            Err(_) => {
                return Err(IndexError::OperationError(
                    "Failed to acquire mutex".to_string(),
                ));
            }
        };

        // Drop the lock before awaiting
        tokio::task::spawn_blocking(move || {
            let rt = tokio::runtime::Handle::current();
            rt.block_on(db.query_with_condition(&condition))
        })
        .await
        .map_err(|e| IndexError::OperationError(format!("Task join error: {}", e)))?
        .map_err(|e| IndexError::OperationError(e.to_string()))
    }

    /// Get a clone of the current configuration
    pub fn get_config(&self) -> IndexConfig {
        self.config.clone()
    }

    /// Index a document with schema validation
    pub async fn index_document_with_validation(
        &self,
        doc: &str,
        doc_id: &str,
        schema: &DocumentSchema,
    ) -> Result<(), IndexError> {
        // Validate the document against the schema
        if let Err(errors) = schema.validate(doc) {
            return Err(IndexError::JsonError(format!(
                "Document validation failed: {}",
                errors.join(", ")
            )));
        }

        // If validation passed, proceed with indexing
        self.index_document(doc, doc_id).await
    }
}

/// Main database structure that manages indexing and querying
#[derive(Clone)] // Add Clone to support ThreadSafeJsonIndexDB
pub struct JsonIndexDB {
    index_partition: PartitionHandle,
    doc_partition: PartitionHandle,
    /// Configuration for this database instance
    config: IndexConfig,
    /// Last validation time for consistency checks
    last_validation: Option<SystemTime>,
}

impl JsonIndexDB {
    /// Create a new jsonindexDB instance
    pub fn new(keyspace: fjall::Keyspace) -> Result<Self> {
        // Use default configuration
        Self::with_config(keyspace, IndexConfig::default())
    }

    /// Create a new JsonIndexDB instance with custom configuration
    pub fn with_config(keyspace: fjall::Keyspace, config: IndexConfig) -> Result<Self> {
        let index_partition = keyspace
            .open_partition(
                &config.index_partition_name,
                PartitionCreateOptions::default(),
            )
            .context(format!(
                "Failed to open index partition '{}'",
                config.index_partition_name
            ))?;

        let doc_partition = keyspace
            .open_partition(
                &config.doc_partition_name,
                PartitionCreateOptions::default(),
            )
            .context(format!(
                "Failed to open document partition '{}'",
                config.doc_partition_name
            ))?;

        Ok(Self {
            index_partition,
            doc_partition,
            config,
            last_validation: None,
        })
    }

    /// Validate the database for consistency
    pub async fn validate(&mut self) -> Result<ValidationReport> {
        let mut report = ValidationReport::new();

        // Placeholder for actual validation logic
        // In a real implementation, this would scan both partitions to ensure
        // that every document reference in the index exists, and every document
        // has proper index entries

        report.add_message("Database validation completed successfully");

        // Update the last validation time
        self.last_validation = Some(SystemTime::now());

        Ok(report)
    }

    /// Convert a JSON object to flattened key-value pairs
    fn flatten_json_object(&self, json_val: &Value) -> Result<Vec<String>> {
        match json_val {
            Value::Object(map) => {
                let flat_entries = flatten_serde_json::flatten(map);
                Ok(flat_entries
                    .into_iter()
                    .map(|(k, v)| {
                        let value_str = match v {
                            Value::String(s) => s,
                            _ => v.to_string(),
                        };
                        format!("{}={}", k, value_str)
                    })
                    .collect())
            }
            _ => Err(anyhow::anyhow!("Root JSON value must be an object")),
        }
    }

    /// Index a document with the given document ID
    pub async fn index_document(&self, doc: &str, doc_id: &str) -> Result<()> {
        let json_val: Value =
            serde_json::from_str(doc).context("Failed to parse document as JSON")?;

        // Store the original document
        self.doc_partition
            .insert(doc_id, doc)
            .context("Failed to store document in doc_partition")?;

        // Flatten the JSON structure for indexing
        let flattened = self.flatten_json_object(&json_val)?;

        // Update the index with flattened key-value pairs
        for key in flattened {
            self.add_to_index(&key, doc_id).await?;
        }

        Ok(())
    }

    /// Add a document ID to an index entry
    async fn add_to_index(&self, key: &str, doc_id: &str) -> Result<()> {
        // Check if the key already exists in the index
        let existing = self
            .index_partition
            .get(key)
            .context(format!("Failed to check if key '{}' exists in index", key))?;

        let doc_ids = match existing {
            Some(ids) => {
                // Deserialize existing array of doc_ids
                let mut ids: Vec<String> = serde_json::from_slice(&ids).context(format!(
                    "Failed to deserialize document IDs for key '{}'",
                    key
                ))?;

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
        let serialized =
            serde_json::to_vec(&doc_ids).context("Failed to serialize document IDs")?;
        self.index_partition
            .insert(key, serialized)
            .context(format!("Failed to update index for key '{}'", key))?;

        Ok(())
    }

    /// Query documents that match the given key-value pattern
    pub async fn query(&self, query: &str) -> Result<Vec<(String, String)>> {
        // Get document IDs that match the query
        let doc_ids = self.get_matching_doc_ids(query).await?;

        // Retrieve the actual documents
        self.get_documents(&doc_ids).await
    }

    /// Complex query with multiple conditions combined with AND/OR logic
    pub async fn complex_query(
        &self,
        queries: &[&str],
        logic: QueryLogic,
    ) -> Result<Vec<(String, String)>> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }

        // Get the first set of results
        let mut result_set = self.get_matching_doc_ids_as_set(queries[0]).await?;

        // Process subsequent queries based on the logic
        for query in &queries[1..] {
            let new_set = self.get_matching_doc_ids_as_set(query).await?;

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
        let result_doc_ids: Vec<String> = result_set.into_iter().collect();
        self.get_documents(&result_doc_ids).await
    }

    /// Get document IDs that match a specific query
    async fn get_matching_doc_ids(&self, query: &str) -> Result<Vec<String>> {
        match self
            .index_partition
            .get(query)
            .context(format!("Failed to query index for '{}'", query))?
        {
            Some(ids_bytes) => {
                let doc_ids: Vec<String> = serde_json::from_slice(&ids_bytes).context(format!(
                    "Failed to deserialize document IDs for query '{}'",
                    query
                ))?;
                Ok(doc_ids)
            }
            None => Ok(Vec::new()),
        }
    }

    /// Get document IDs that match a specific query as a HashSet
    async fn get_matching_doc_ids_as_set(&self, query: &str) -> Result<HashSet<String>> {
        match self
            .index_partition
            .get(query)
            .context(format!("Failed to query index for '{}'", query))?
        {
            Some(ids_bytes) => {
                let doc_ids: Vec<String> = serde_json::from_slice(&ids_bytes).context(format!(
                    "Failed to deserialize document IDs for query '{}'",
                    query
                ))?;
                Ok(doc_ids.into_iter().collect())
            }
            None => Ok(HashSet::new()),
        }
    }

    /// Retrieve documents by their IDs
    async fn get_documents(&self, doc_ids: &[String]) -> Result<Vec<(String, String)>> {
        let mut results = Vec::with_capacity(doc_ids.len());

        for doc_id in doc_ids {
            if let Some(doc_bytes) = self
                .doc_partition
                .get(doc_id)
                .context(format!("Failed to retrieve document with ID '{}'", doc_id))?
            {
                let doc_str = String::from_utf8(doc_bytes.to_vec()).context(format!(
                    "Failed to convert document '{}' to UTF-8 string",
                    doc_id
                ))?;
                results.push((doc_id.clone(), doc_str));
            }
        }

        Ok(results)
    }

    /// Delete a document and its index entries
    pub async fn delete_document(&self, doc_id: &str) -> Result<()> {
        // First get the document
        let doc_bytes = match self
            .doc_partition
            .get(doc_id)
            .context(format!("Failed to check if document '{}' exists", doc_id))?
        {
            Some(bytes) => bytes,
            None => return Ok(()), // Document doesn't exist, nothing to delete
        };

        // Parse the document to recreate the index keys
        let doc_str = String::from_utf8(doc_bytes.to_vec()).context(format!(
            "Failed to convert document '{}' to UTF-8 string",
            doc_id
        ))?;
        let json_val: Value = serde_json::from_str(&doc_str)
            .context(format!("Failed to parse document '{}' as JSON", doc_id))?;

        // Flatten the JSON structure to get all the keys we need to update
        let flattened = self.flatten_json_object(&json_val)?;

        // Remove this document ID from all relevant index entries
        for key in flattened {
            self.remove_from_index(&key, doc_id).await?;
        }

        // Finally, delete the document itself
        self.doc_partition
            .remove(doc_id)
            .context(format!("Failed to delete document '{}'", doc_id))?;

        Ok(())
    }

    /// Remove a document ID from an index entry
    async fn remove_from_index(&self, key: &str, doc_id: &str) -> Result<()> {
        if let Some(ids_bytes) = self
            .index_partition
            .get(key)
            .context(format!("Failed to get index entries for key '{}'", key))?
        {
            let mut doc_ids: Vec<String> = serde_json::from_slice(&ids_bytes).context(format!(
                "Failed to deserialize document IDs for key '{}'",
                key
            ))?;

            // Remove the doc_id if it exists
            doc_ids.retain(|id| id != doc_id);

            if doc_ids.is_empty() {
                // If no documents left, remove the index entry entirely
                self.index_partition.remove(key).context(format!(
                    "Failed to remove empty index entry for key '{}'",
                    key
                ))?;
            } else {
                // Update the index with the remaining document IDs
                let serialized =
                    serde_json::to_vec(&doc_ids).context("Failed to serialize document IDs")?;
                self.index_partition
                    .insert(key, serialized)
                    .context(format!("Failed to update index for key '{}'", key))?;
            }
        }

        Ok(())
    }

    /// Batch index multiple documents with their document IDs
    pub async fn batch_index_documents(&self, docs: &[(&str, &str)]) -> Result<()> {
        for (doc, doc_id) in docs {
            self.index_document(doc, doc_id).await?;
        }
        Ok(())
    }

    /// Batch delete multiple documents by their IDs
    pub async fn batch_delete_documents(&self, doc_ids: &[&str]) -> Result<()> {
        for doc_id in doc_ids {
            self.delete_document(doc_id).await?;
        }
        Ok(())
    }

    /// Batch query multiple patterns and return combined results
    pub async fn batch_query(
        &self,
        queries: &[&str],
        logic: QueryLogic,
    ) -> Result<Vec<(String, String)>> {
        self.complex_query(queries, logic).await
    }

    /// Query documents using a strongly typed query condition
    pub async fn query_with_condition(
        &self,
        condition: &QueryCondition,
    ) -> Result<Vec<(String, String)>> {
        match condition.operator {
            ComparisonOperator::Equals => {
                // Direct index lookup for equals
                self.query(&condition.to_index_key()).await
            }
            // For other operators, we need to fetch all documents for the field and filter
            _ => {
                // This is a simplistic implementation for demonstration
                // In a real system, you'd want more efficient filtering
                let field_prefix = format!("{}=", condition.field);
                let all_keys = self.get_keys_with_prefix(&field_prefix).await?;

                let mut matching_doc_ids = HashSet::new();
                for key in all_keys {
                    let doc_ids = self.get_matching_doc_ids(&key).await?;

                    for doc_id in doc_ids {
                        if self.document_matches_condition(&doc_id, condition).await? {
                            matching_doc_ids.insert(doc_id);
                        }
                    }
                }

                let result_doc_ids: Vec<String> = matching_doc_ids.into_iter().collect();
                self.get_documents(&result_doc_ids).await
            }
        }
    }

    /// Check if a specific document matches the given condition
    async fn document_matches_condition(
        &self,
        doc_id: &str,
        condition: &QueryCondition,
    ) -> Result<bool> {
        let doc_bytes = match self.doc_partition.get(doc_id)? {
            Some(bytes) => bytes,
            None => return Ok(false),
        };

        let doc_str = String::from_utf8(doc_bytes.to_vec())?;
        let json_val: Value = serde_json::from_str(&doc_str)?;

        // Navigate through the nested structure to find the field
        let field_parts: Vec<&str> = condition.field.split('.').collect();
        let mut current_val = &json_val;

        for part in &field_parts {
            match current_val {
                Value::Object(map) => {
                    if let Some(val) = map.get(*part) {
                        current_val = val;
                    } else {
                        return Ok(false); // Field not found
                    }
                }
                _ => return Ok(false), // Not an object, can't navigate further
            }
        }

        // Apply the comparison operator
        match condition.operator {
            ComparisonOperator::Equals => {
                let field_str = match current_val {
                    Value::String(s) => s.clone(),
                    _ => current_val.to_string(),
                };
                Ok(field_str == condition.value)
            }
            ComparisonOperator::Contains => {
                let field_str = match current_val {
                    Value::String(s) => s.clone(),
                    _ => current_val.to_string(),
                };
                Ok(field_str.contains(&condition.value))
            }
            ComparisonOperator::GreaterThan => match current_val {
                Value::Number(n) if n.is_f64() => {
                    if let Ok(val) = condition.value.parse::<f64>() {
                        Ok(n.as_f64().unwrap() > val)
                    } else {
                        Ok(false)
                    }
                }
                Value::Number(n) if n.is_i64() => {
                    if let Ok(val) = condition.value.parse::<i64>() {
                        Ok(n.as_i64().unwrap() > val)
                    } else {
                        Ok(false)
                    }
                }
                Value::Number(n) if n.is_u64() => {
                    if let Ok(val) = condition.value.parse::<u64>() {
                        Ok(n.as_u64().unwrap() > val)
                    } else {
                        Ok(false)
                    }
                }
                Value::String(s) => Ok(s > &condition.value),
                _ => Ok(false),
            },
            ComparisonOperator::LessThan => match current_val {
                Value::Number(n) if n.is_f64() => {
                    if let Ok(val) = condition.value.parse::<f64>() {
                        Ok(n.as_f64().unwrap() < val)
                    } else {
                        Ok(false)
                    }
                }
                Value::Number(n) if n.is_i64() => {
                    if let Ok(val) = condition.value.parse::<i64>() {
                        Ok(n.as_i64().unwrap() < val)
                    } else {
                        Ok(false)
                    }
                }
                Value::Number(n) if n.is_u64() => {
                    if let Ok(val) = condition.value.parse::<u64>() {
                        Ok(n.as_u64().unwrap() < val)
                    } else {
                        Ok(false)
                    }
                }
                Value::String(s) => Ok(s < &condition.value),
                _ => Ok(false),
            },
        }
    }

    /// Get all keys in index_partition with a specific prefix
    async fn get_keys_with_prefix(&self, prefix: &str) -> Result<Vec<String>> {
        // This is a simplified implementation
        // In a real system, you'd use a range scan or prefix iterator
        // but that depends on the capabilities of the underlying store (fjall)

        // For demonstration purposes, assume we have a method to list all keys
        // and filter those that start with our prefix
        // This would be inefficient for large datasets
        let all_keys = vec!["example.key=value".to_string()]; // Placeholder
        let matching_keys: Vec<String> = all_keys
            .into_iter()
            .filter(|k| k.starts_with(prefix))
            .collect();

        Ok(matching_keys)
    }

    /// Index a document with a strongly typed document ID
    pub async fn index_document_typed(
        &self,
        doc: &str,
        doc_id: &DocumentId,
    ) -> Result<(), IndexError> {
        self.index_document(doc, doc_id.as_str())
            .await
            .map_err(|e| IndexError::OperationError(format!("Failed to index document: {}", e)))
    }

    /// Delete a document with a strongly typed document ID
    pub async fn delete_document_typed(&self, doc_id: &DocumentId) -> Result<(), IndexError> {
        self.delete_document(doc_id.as_str())
            .await
            .map_err(|e| IndexError::OperationError(format!("Failed to delete document: {}", e)))
    }

    /// Get documents by their strongly typed IDs
    pub async fn get_documents_typed(
        &self,
        doc_ids: &[DocumentId],
    ) -> Result<Vec<(DocumentId, String)>, IndexError> {
        let string_ids: Vec<String> = doc_ids.iter().map(|id| id.as_str().to_string()).collect();

        let results = self.get_documents(&string_ids).await.map_err(|e| {
            IndexError::OperationError(format!("Failed to retrieve documents: {}", e))
        })?;

        let typed_results: Vec<(DocumentId, String)> = results
            .into_iter()
            .map(|(id, doc)| (DocumentId(id), doc))
            .collect();

        Ok(typed_results)
    }
}

/// A report of database validation results
#[derive(Debug, Default)]
pub struct ValidationReport {
    /// Messages generated during validation
    messages: Vec<String>,
    /// Number of issues found
    issue_count: usize,
    /// Time when validation was completed
    completed_at: Option<SystemTime>,
}

impl ValidationReport {
    /// Create a new validation report
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            issue_count: 0,
            completed_at: None,
        }
    }

    /// Add a message to the report
    pub fn add_message(&mut self, message: &str) {
        self.messages.push(message.to_string());
    }

    /// Add an issue to the report
    pub fn add_issue(&mut self, issue: &str) {
        self.add_message(&format!("ISSUE: {}", issue));
        self.issue_count += 1;
    }

    /// Mark the validation as complete
    pub fn complete(&mut self) {
        self.completed_at = Some(SystemTime::now());
    }

    /// Get the number of issues found
    pub fn issue_count(&self) -> usize {
        self.issue_count
    }

    /// Get all messages from the report
    pub fn messages(&self) -> &[String] {
        &self.messages
    }
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Validation Report ===")?;
        writeln!(f, "Issues found: {}", self.issue_count)?;

        if let Some(time) = self.completed_at {
            if let Ok(duration) = time.duration_since(SystemTime::UNIX_EPOCH) {
                writeln!(
                    f,
                    "Completed at: {} seconds since epoch",
                    duration.as_secs()
                )?;
            }
        }

        writeln!(f, "--- Messages ---")?;
        for message in &self.messages {
            writeln!(f, "  {}", message)?;
        }

        Ok(())
    }
}

/// Document schema for validation
#[derive(Debug, Clone)]
pub struct DocumentSchema {
    /// Required fields that must be present
    required_fields: Vec<String>,
    /// Field types for validation
    field_types: std::collections::HashMap<String, FieldType>,
    /// Schema name
    name: String,
}

/// Field types for schema validation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FieldType {
    /// String value
    String,
    /// Numeric value
    Number,
    /// Boolean value
    Boolean,
    /// Object/map value
    Object,
    /// Array value
    Array,
    /// Any value type
    Any,
}

impl DocumentSchema {
    /// Create a new document schema
    pub fn new(name: &str) -> Self {
        Self {
            required_fields: Vec::new(),
            field_types: std::collections::HashMap::new(),
            name: name.to_string(),
        }
    }

    /// Add a required field to the schema
    pub fn add_required_field(&mut self, field: &str, field_type: FieldType) -> &mut Self {
        self.required_fields.push(field.to_string());
        self.field_types.insert(field.to_string(), field_type);
        self
    }

    /// Add an optional field to the schema
    pub fn add_optional_field(&mut self, field: &str, field_type: FieldType) -> &mut Self {
        self.field_types.insert(field.to_string(), field_type);
        self
    }

    /// Validate a document against this schema
    pub fn validate(&self, doc: &str) -> Result<(), Vec<String>> {
        let json_val: Result<Value, _> = serde_json::from_str(doc);

        let json_val = match json_val {
            Ok(val) => val,
            Err(e) => return Err(vec![format!("Invalid JSON: {}", e)]),
        };

        let mut errors = Vec::new();

        // Check required fields
        for field in &self.required_fields {
            let field_parts: Vec<&str> = field.split('.').collect();
            let mut current_val = &json_val;
            let mut field_found = true;

            for part in &field_parts {
                match current_val {
                    Value::Object(map) => {
                        if let Some(val) = map.get(*part) {
                            current_val = val;
                        } else {
                            field_found = false;
                            break;
                        }
                    }
                    _ => {
                        field_found = false;
                        break;
                    }
                }
            }

            if !field_found {
                errors.push(format!("Required field '{}' is missing", field));
            } else {
                // Check field type
                if let Some(expected_type) = self.field_types.get(field) {
                    if !self.check_field_type(current_val, *expected_type) {
                        errors.push(format!(
                            "Field '{}' has wrong type, expected {:?}",
                            field, expected_type
                        ));
                    }
                }
            }
        }

        // Check optional field types if they exist
        for (field, expected_type) in &self.field_types {
            if self.required_fields.contains(field) {
                // Already checked
                continue;
            }

            let field_parts: Vec<&str> = field.split('.').collect();
            let mut current_val = &json_val;
            let mut field_found = true;

            for part in &field_parts {
                match current_val {
                    Value::Object(map) => {
                        if let Some(val) = map.get(*part) {
                            current_val = val;
                        } else {
                            field_found = false;
                            break;
                        }
                    }
                    _ => {
                        field_found = false;
                        break;
                    }
                }
            }

            if field_found && !self.check_field_type(current_val, *expected_type) {
                errors.push(format!(
                    "Optional field '{}' has wrong type, expected {:?}",
                    field, expected_type
                ));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Check if a value matches the expected field type
    fn check_field_type(&self, value: &Value, expected_type: FieldType) -> bool {
        match expected_type {
            FieldType::String => value.is_string(),
            FieldType::Number => value.is_number(),
            FieldType::Boolean => value.is_boolean(),
            FieldType::Object => value.is_object(),
            FieldType::Array => value.is_array(),
            FieldType::Any => true,
        }
    }
}

impl JsonIndexDB {
    /// Index a document with schema validation
    pub async fn index_document_with_validation(
        &self,
        doc: &str,
        doc_id: &str,
        schema: &DocumentSchema,
    ) -> Result<()> {
        // Validate the document against the schema
        if let Err(errors) = schema.validate(doc) {
            return Err(anyhow::anyhow!(
                "Document validation failed for schema '{}': {}",
                schema.name,
                errors.join(", ")
            ));
        }

        // If validation passed, proceed with indexing
        self.index_document(doc, doc_id).await
    }
}

#[cfg(test)]
mod tests {
    use fjall::Config;

    use super::*;
    use std::fs;
    use std::path::Path;

    #[tokio::test]
    async fn test_basic_flow() -> Result<()> {
        // Create a temporary directory for testing
        let test_dir = "test_db_data";
        if Path::new(test_dir).exists() {
            fs::remove_dir_all(test_dir)?;
        }

        // Create a database instance
        let keyspace = Config::new(test_dir).open()?;
        let json_index = JsonIndexDB::new(keyspace)?;

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
        json_index.index_document(doc, doc_id).await?;

        // Test simple query
        let results = json_index.query("name=Test User").await?;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, doc_id);

        // Test complex AND query - both conditions match
        let results = json_index
            .complex_query(&["name=Test User", "age=25"], QueryLogic::And)
            .await?;
        assert_eq!(results.len(), 1);

        // Test complex AND query - one condition doesn't match
        let results = json_index
            .complex_query(&["name=Test User", "age=30"], QueryLogic::And)
            .await?;
        assert_eq!(results.len(), 0);

        // Test complex OR query
        let results = json_index
            .complex_query(&["age=25", "age=30"], QueryLogic::Or)
            .await?;
        assert_eq!(results.len(), 1);

        // Test nested field query
        let results = json_index.query("address.city=Test City").await?;
        assert_eq!(results.len(), 1);

        // Test deletion
        json_index.delete_document(doc_id).await?;
        let results = json_index.query("name=Test User").await?;
        assert_eq!(results.len(), 0);

        // Clean up test directory
        fs::remove_dir_all(test_dir)?;

        Ok(())
    }

    #[tokio::test]
    async fn test_batch_operations() -> Result<()> {
        // Create a temporary directory for testing
        let test_dir = "test_batch_ops";
        if Path::new(test_dir).exists() {
            fs::remove_dir_all(test_dir)?;
        }

        // Create a database instance
        let keyspace = Config::new(test_dir).open()?;
        let json_index = JsonIndexDB::new(keyspace)?;

        // Test documents
        let doc1 = r#"{"name": "User 1", "age": 25}"#;
        let doc2 = r#"{"name": "User 2", "age": 30}"#;
        let doc3 = r#"{"name": "User 3", "age": 35}"#;

        // Batch index documents
        let docs = vec![(doc1, "doc1"), (doc2, "doc2"), (doc3, "doc3")];

        json_index.batch_index_documents(&docs).await?;

        // Test batch query
        let results = json_index
            .batch_query(&["age=25", "age=30"], QueryLogic::Or)
            .await?;
        assert_eq!(results.len(), 2);

        // Test batch delete
        json_index.batch_delete_documents(&["doc1", "doc2"]).await?;

        let results = json_index.query("name=User 1").await?;
        assert_eq!(results.len(), 0);

        let results = json_index.query("name=User 3").await?;
        assert_eq!(results.len(), 1);

        // Clean up test directory
        fs::remove_dir_all(test_dir)?;

        Ok(())
    }

    #[tokio::test]
    async fn test_query_conditions() -> Result<()> {
        // Create a temporary directory for testing
        let test_dir = "test_query_conditions";
        if Path::new(test_dir).exists() {
            fs::remove_dir_all(test_dir)?;
        }

        // Create a database instance
        let keyspace = Config::new(test_dir).open()?;
        let json_index = JsonIndexDB::new(keyspace)?;

        // Test document
        let doc = r#"{
            "name": "Test User",
            "age": 25,
            "tags": ["json", "database", "rust"],
            "score": 9.5
        }"#;

        // Index the document
        let doc_id = "test-doc-123";
        json_index.index_document(doc, doc_id).await?;

        // Test different condition types
        let equals_condition = QueryCondition::new("name", ComparisonOperator::Equals, "Test User");
        let results = json_index.query_with_condition(&equals_condition).await?;
        assert_eq!(results.len(), 1);

        // Clean up test directory
        fs::remove_dir_all(test_dir)?;

        Ok(())
    }

    #[tokio::test]
    async fn test_typed_document_id() -> Result<()> {
        // Create a temporary directory for testing
        let test_dir = "test_typed_id";
        if Path::new(test_dir).exists() {
            fs::remove_dir_all(test_dir)?;
        }

        // Create a database instance
        let keyspace = Config::new(test_dir).open()?;
        let json_index = JsonIndexDB::new(keyspace)?;

        // Test document
        let doc = r#"{"name": "Test User", "age": 25}"#;

        // Test typed document ID
        let typed_id = DocumentId::new("typed-123").map_err(|e| anyhow::anyhow!("{}", e))?;

        // Should succeed
        json_index
            .index_document_typed(doc, &typed_id)
            .await
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        // Test empty ID validation
        let empty_id_result = DocumentId::new("");
        assert!(empty_id_result.is_err());

        // Clean up test directory
        fs::remove_dir_all(test_dir)?;

        Ok(())
    }

    #[tokio::test]
    async fn test_custom_config() -> Result<()> {
        // Create a temporary directory for testing
        let test_dir = "test_custom_config";
        if Path::new(test_dir).exists() {
            fs::remove_dir_all(test_dir)?;
        }

        // Create a custom configuration
        let config = IndexConfig {
            case_sensitive: false,
            normalize_strings: true,
            max_results: 500,
            index_partition_name: "custom_index".to_string(),
            doc_partition_name: "custom_docs".to_string(),
        };

        // Create a database instance with custom config
        let keyspace = Config::new(test_dir).open()?;
        let json_index = JsonIndexDB::with_config(keyspace, config)?;

        // Test document
        let doc = r#"{"name": "Test User", "age": 25}"#;
        let doc_id = "test-doc-123";

        // Basic operation should work with custom config
        json_index.index_document(doc, doc_id).await?;
        let results = json_index.query("name=Test User").await?;
        assert_eq!(results.len(), 1);

        // Clean up test directory
        fs::remove_dir_all(test_dir)?;

        Ok(())
    }

    #[tokio::test]
    async fn test_document_validation() -> Result<()> {
        // Create a temporary directory for testing
        let test_dir = "test_validation";
        if Path::new(test_dir).exists() {
            fs::remove_dir_all(test_dir)?;
        }

        // Create a database instance
        let keyspace = Config::new(test_dir).open()?;
        let json_index = JsonIndexDB::new(keyspace)?;

        // Define a schema
        let mut schema = DocumentSchema::new("user");
        schema
            .add_required_field("name", FieldType::String)
            .add_required_field("age", FieldType::Number)
            .add_optional_field("email", FieldType::String);

        // Valid document
        let valid_doc = r#"{"name": "Test User", "age": 25, "email": "test@example.com"}"#;
        // Should index successfully
        json_index
            .index_document_with_validation(valid_doc, "valid-doc", &schema)
            .await?;

        // Invalid document (missing required field)
        let invalid_doc1 = r#"{"name": "Test User"}"#;
        let result1 = json_index
            .index_document_with_validation(invalid_doc1, "invalid-doc1", &schema)
            .await;
        assert!(result1.is_err());

        // Invalid document (wrong field type)
        let invalid_doc2 = r#"{"name": "Test User", "age": "twenty five"}"#;
        let result2 = json_index
            .index_document_with_validation(invalid_doc2, "invalid-doc2", &schema)
            .await;
        assert!(result2.is_err());

        // Clean up test directory
        fs::remove_dir_all(test_dir)?;

        Ok(())
    }

    #[tokio::test]
    async fn test_validation() -> Result<()> {
        // Create a temporary directory for testing
        let test_dir = "test_db_validation";
        if Path::new(test_dir).exists() {
            fs::remove_dir_all(test_dir)?;
        }

        // Create a database instance
        let keyspace = Config::new(test_dir).open()?;
        let mut json_index = JsonIndexDB::new(keyspace)?;

        // Add some documents
        let doc1 = r#"{"name": "User 1", "age": 25}"#;
        let doc2 = r#"{"name": "User 2", "age": 30}"#;

        json_index.index_document(doc1, "doc1").await?;
        json_index.index_document(doc2, "doc2").await?;

        // Run validation (this needs the mutable reference)
        let validation_report = json_index.validate().await?;

        // Verify validation succeeded
        assert_eq!(validation_report.issue_count(), 0);

        // Clean up test directory
        fs::remove_dir_all(test_dir)?;

        Ok(())
    }

    #[tokio::test]
    async fn test_transactions() -> Result<()> {
        // Create a temporary directory for testing
        let test_dir = "test_transactions";
        if Path::new(test_dir).exists() {
            fs::remove_dir_all(test_dir)?;
        }

        // Create a thread-safe database instance
        let keyspace = Config::new(test_dir).open()?;
        let db = ThreadSafeJsonIndexDB::new(keyspace)?;

        // Begin a transaction
        let mut txn = db.begin_transaction();

        // Add operations to the transaction
        let doc1 = r#"{"name": "User 1", "age": 25}"#;
        let doc2 = r#"{"name": "User 2", "age": 30}"#;

        txn.index_document(doc1, "doc1")
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        txn.index_document(doc2, "doc2")
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        // Commit the transaction
        db.commit_transaction(txn)
            .await
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        // Verify both documents were indexed
        let results1 = db
            .query("name=User 1")
            .await
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let results2 = db
            .query("name=User 2")
            .await
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        assert_eq!(results1.len(), 1);
        assert_eq!(results2.len(), 1);

        // Test rollback
        let mut txn2 = db.begin_transaction();
        txn2.delete_document("doc1")
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        // Rollback instead of committing
        db.rollback_transaction(txn2)
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        // Verify doc1 still exists
        let results = db
            .query("name=User 1")
            .await
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        assert_eq!(results.len(), 1);

        // Clean up test directory
        fs::remove_dir_all(test_dir)?;

        Ok(())
    }

    #[tokio::test]
    async fn test_concurrency() -> Result<()> {
        use std::sync::Arc;
        use tokio::task;

        // Create a temporary directory for testing
        let test_dir = "test_concurrency";
        if Path::new(test_dir).exists() {
            fs::remove_dir_all(test_dir)?;
        }

        // Create a thread-safe database instance
        let keyspace = Config::new(test_dir).open()?;
        let db = Arc::new(ThreadSafeJsonIndexDB::new(keyspace)?);

        // Test concurrent indexing
        let mut handles = Vec::new();

        for i in 0..10 {
            let db_clone = db.clone();
            let handle = task::spawn(async move {
                let doc = format!(r#"{{"name": "User {}", "age": {}}}"#, i, 20 + i);
                let doc_id = format!("doc{}", i);
                db_clone.index_document(&doc, &doc_id).await
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            handle
                .await
                .unwrap()
                .map_err(|e| anyhow::anyhow!("{}", e))?;
        }

        // Verify all documents were indexed
        let results = db
            .complex_query(
                &[
                    "age=20", "age=21", "age=22", "age=23", "age=24", "age=25", "age=26", "age=27",
                    "age=28", "age=29",
                ],
                QueryLogic::Or,
            )
            .await
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        assert_eq!(results.len(), 10);

        // Clean up test directory
        fs::remove_dir_all(test_dir)?;

        Ok(())
    }
}
