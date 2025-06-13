use std::sync::Arc;

use tokio::task::JoinHandle;

use crate::indexer::index_writer::error_in_index_worker_thread;
use crate::indexer::operation::AddOperation;
use crate::indexer::segment_updater::save_metas;
use crate::indexer::{SegmentWriter, TOKIO_RUNTIME};
use crate::{Directory, Document, Index, IndexMeta, Segment};

#[doc(hidden)]
pub struct SingleSegmentIndexWriter {
    segment: Segment,
    tx: Arc<async_channel::Sender<Document>>,
    join_handle: JoinHandle<crate::Result<SegmentWriter>>,
}

impl SingleSegmentIndexWriter {
    pub fn new(index: Index, mem_budget: usize) -> crate::Result<Self> {
        let segment = index.new_segment();
        let mut segment_writer = SegmentWriter::for_segment(mem_budget, segment.clone())?;
        let (tx, rx) = async_channel::unbounded();
        let join_handle = TOKIO_RUNTIME.spawn(async move {
            let mut opstamp = 0;
            while let Ok(document) = rx.recv().await {
                segment_writer
                    .add_document(AddOperation { opstamp, document })
                    .await?;
                opstamp += 1;
            }
            Ok(segment_writer)
        });
        Ok(Self {
            segment,
            tx: Arc::new(tx),
            join_handle,
        })
    }

    pub fn add_document(&mut self, document: Document) -> crate::Result<()> {
        let tx = self.tx.clone();
        if TOKIO_RUNTIME
            .block_on(async move { tx.send(document).await })
            .is_ok()
        {
            return Ok(());
        }
        Err(error_in_index_worker_thread(
            "An index writer encounter erros.",
        ))
    }

    pub fn finalize(self) -> crate::Result<Index> {
        TOKIO_RUNTIME.block_on(async {
            self.tx.close();
            let segment_writer = self
                .join_handle
                .await
                .map_err(|_| error_in_index_worker_thread("Worker thread panicked."))?
                .map_err(|_| error_in_index_worker_thread("Worker thread failed."))?;

            let max_doc = segment_writer.max_doc();
            segment_writer.finalize().await?;
            let segment: Segment = self.segment.with_max_doc(max_doc);
            let index = segment.index();
            let index_meta = IndexMeta {
                index_settings: index.settings().clone(),
                segments: vec![segment.meta().clone()],
                schema: index.schema(),
                opstamp: 0,
                payload: None,
            };
            save_metas(&index_meta, index.directory())?;
            index.directory().sync_directory()?;
            Ok(segment.index().clone())
        })
    }
}
