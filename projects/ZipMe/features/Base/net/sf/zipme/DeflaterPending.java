

package net.sf.zipme;


class DeflaterPending extends PendingBuffer {
  public DeflaterPending(){
    super(DeflaterConstants.PENDING_BUF_SIZE);
  }
}
