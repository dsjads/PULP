

package net.sf.zipme;
import java.io.IOException;
import java.io.OutputStream;
import java.io.UnsupportedEncodingException;
import java.util.Enumeration;
import java.util.Vector;


public class ZipOutputStream extends DeflaterOutputStream implements ZipConstants {

  public static final int STORED=0;

  public static final int DEFLATED=8;

  public ZipOutputStream(  OutputStream out){
    super(out,new Deflater(Deflater.DEFAULT_COMPRESSION,true));
  }
}
